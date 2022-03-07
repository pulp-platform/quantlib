import torch
from torch import nn

from typing import Optional

from copy import deepcopy

from parse import parse

from functools import partial

import numpy as np

from quantlib.editing.lightweight import LightweightNode, LightweightGraph
from quantlib.editing.fx.passes.pact import PACTInclusiveTracer
from quantlib.editing.fx.passes import FindSequentialPatternsPass
from quantlib.algorithms import Controller
from quantlib.algorithms.pact import PACTConv1d, PACTConv2d
from quantlib.editing.lightweight.rules.filters import SubTypeFilter


class PROFITController(Controller):
    #def __init__(self, modules : list, schedule : dict, net_trace : callable = PACTInclusiveTracer, pattern_trace : callable = PACTInclusiveTracer, patterns : Optional[list] = None, verbose : bool = False):
    def __init__(self, nodes : list, schedule : dict, verbose : bool = False):
        self.nodes = nodes
        self.schedule = {int(k): v for k, v in schedule.items()}
        #self.trace = trace
        # if patterns is None:
        #     self.patterns = []
        #     self.patterns.append(nn.Sequential(PACTConv2d(1,1,1), nn.BatchNorm2d(1)))
        #     self.patterns.append(nn.Sequential(PACTConv1d(1,1,1), nn.BatchNorm1d(1)))
        # else:
        #     self.patterns = patterns
        self.verbose = verbose
        self.sampling = False
        self.stats = {v.name:{'mean':torch.zeros([v.module.out_channels,]), 'std':torch.ones([v.module.out_channels])} for v in self.nodes}
        self.metric_map = {k:0. for k in self.stats.keys()}
        self.hooks = []

    @staticmethod
    def get_modules(net):
        # we only care about PACTConvNd's
        nl = LightweightGraph.build_nodes_list(net)
        conv_filter = SubTypeFilter(PACTConv1d) | SubTypeFilter(PACTConv2d)
        conv_nodes = conv_filter(nl)
        # we need names attached to the modules so we can store the statistics
        # in a serializable dict
        return conv_nodes


    def sample_hook(self, module, input, output, module_name):
        old_mean = self.stats[module_name]['mean']
        old_std = self.stats[module_name]['std']
        out_channels = torch.transpose(output.clone().detach().cpu(), 0, 1).contiguous().view(module.out_channels, -1)
        mean = torch.mean(out_channels, 1)
        std = torch.std(out_channels, 1)
        # only consider channels where neither the old nor the standard
        # distribution is not 0
        valid_channels = (std > 1e-8) & (old_std > 1e-8)
        mean_clean = mean[valid_channels]
        old_mean_clean = old_mean[valid_channels]
        std_clean = std[valid_channels]
        old_std_clean = old_std[valid_channels]
        # taken from https://github.com/EunhyeokPark/PROFIT/blob/master/my_lib/train_test.py
        aiwq_metric = torch.log(old_std_clean/std_clean) + \
            (std_clean ** 2 + (mean_clean - old_mean_clean) ** 2) / (2 * old_std_clean ** 2) - 0.5

        m = aiwq_metric.mean().cpu().numpy()
        if m <= 1.:
            self.metric_map[module_name] = self.metric_map[module_name] * 0.999 + 0.001 * m
        self.stats[module_name]['mean'] = mean
        self.stats[module_name]['std'] = std

    def step_pre_training_epoch(self, epoch, *args, **kwargs):
        if epoch in self.schedule.keys():
            cur_cmds = self.schedule[epoch]
            if not isinstance(cur_cmds, list):
                cur_cmds = [cur_cmds]
            for cmd in cur_cmds:
                self.log(f"Epoch {epoch} - running command {cmd}")
                if cmd == "start_sample":
                    self.sampling = True
                    for n in self.nodes:
                        self.hooks.append(n.module.register_forward_hook(partial(self.sample_hook, module_name=n.name)))
                elif cmd == "stop_sample":
                    for h in self.hooks:
                        h.remove()
                    self.hooks = []
                    self.sampling = False
                elif cmd[:6] == "freeze":
                    res = parse("freeze {}", cmd)
                    frac = float(res.fixed[0])
                    assert frac <= 1. and frac >= 0., f"PROFITController, epoch {epoch}: Invalid freeze fraction {frac}!"
                    mm_keys_sorted = sorted(list(self.metric_map.keys()), key=lambda k:self.metric_map[k], reverse=True)
                    n_modules = len(mm_keys_sorted)
                    n_to_freeze = int(np.ceil(n_modules*frac))
                    for n in self.nodes[:n_to_freeze]:
                        n.module.freeze_params()
                    # unfreeze modules which should not be frozen - this allows
                    # to have non-monotonically increasing fractions 
                    for n in self.nodes[n_to_freeze:]:
                        n.module.unfreeze_params()

                elif cmd == "start_verbose":
                    self.verbose = True
                else:
                    assert cmd == "stop_verbose", f"PROFITController got invalid command {cmd}"
                    self.verbose = False

    def step_pre_validation_epoch(self, epoch, *args, **kwargs):
        pass

    def log(self, msg : str):
        if self.verbose:
            print("[PROFITController]   ", msg)




    def state_dict(self):
        return {'verbose' : self.verbose, 'sampling' : self.sampling, 'stats' : self.stats, 'metric_map': self.metric_map}

    def load_state_dict(self, state_dict : dict):
        self.verbose = state_dict['verbose']
        self.sampling = state_dict['sampling']
        self.stats = state_dict['stats']
        self.metric_map = state_dict['metric_map']
