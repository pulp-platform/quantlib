from typing import Optional

import torch
import json

import numpy as np

from torch import nn, fx

from quantlib.algorithms.bb import BBAct, BBConv2d, BBLinear, _BB_LINOPS
from quantlib.algorithms import Controller
from quantlib.editing.lightweight import LightweightNode


def madd_of_conv(insize, c):
    madds_per_pixel = c.kernel_size[0]*c.kernel_size[1]*c.in_channels*c.out_channels//c.groups
    return madds_per_pixel * insize * insize//(c.stride[0]*c.stride[1])
def madd_of_lin(l):
    return l.in_features * l.out_features

def fx_node_of_ql_node(gm : fx.GraphModule, n : LightweightNode):
    for fxn in gm.graph.nodes:
        if fxn.op == "call_module" and fxn.target == n.name:
            return fxn
    return None

class BBGateController:
    # A controller which calculates the loss term for a given BB layer
    # according to its complexity. The complexity is not calculated by the
    # controller itself, but instead by the entity attaching it to the layer
    # and is encoded in the "loss_scale" parameter.
    def __init__(self, layer : nn.Module, loss_scale : Optional[float] = None, gate_init : float = 2.):
        # 0. Sanity check
        assert isinstance(layer, (BBConv2d, BBLinear, BBAct)), "Layer of BBGateController must be BBAct, BBLinear, BBConv1d or BBConv2d!"

        # 1. set up fields
        self.layer = layer
        self.loss_scale = 1. if loss_scale is None else loss_scale

        # 2. set up gating parameters of conv and act layer
        n_precs = len(layer.precs)

        assert "bb_gates" not in dict(layer.named_parameters()), "Expected linear layer to not have 'bb_gates' parameter!"
        gates = nn.Parameter(torch.full((n_precs-1,), gate_init))
        layer.bb_gates = gates
        layer.register_gate_ctrl(self)
        self.gates = gates
        # for conv layers, this contains the name of the preceding activation layer
        self.linked_layer = None

    def loss_term(self):
        ccdfs = self.layer.ccdf0()
        # ehehe cumprod :^)
        cumulative_ccdfs = torch.cumprod(ccdfs, 0)
        gated_precs = torch.tensor(self.layer.precs[1:], device=cumulative_ccdfs.device)
        loss_term = torch.sum(cumulative_ccdfs * gated_precs)

        return self.loss_scale * loss_term


class BBExportController(Controller):
    # a controller which exports an info .json about a BB-quantized network.
    # The .json contains the number of levels for each layer as well as info
    # such as the total number of BOPs and equivalent number of bits
    def __init__(self, nodes_list : list, export_file : str, input_bitwidth : int = 8, net : Optional[fx.GraphModule] = None):
        self.nodes_list = nodes_list
        self.export_file = export_file
        self.input_bitwidth = input_bitwidth
        # if `net` is an instance of DataParallel, we have to strip 'module'
        # from node names
        net_is_dp = False
        if isinstance(net, nn.DataParallel):
            net_is_dp = True
            net = net.module
        self.net_is_dp = net_is_dp
        self.net = net

    @staticmethod
    def get_modules(*args, **kwargs):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def step_pre_training_epoch(self, *args, **kwargs):
        pass

    def step_pre_training_batch(self, *args, **kwargs):
        pass

    def fix_node_name(self, name : str):
        if self.net_is_dp:
            return name.lstrip('module.')
        return name

    def step_pre_validation_epoch(self, *args, **kwargs):
        prec_dict = {}
        tot_bops = 0
        bops_8b = 0
        for n in self.nodes_list:
            module_levels = n.module.get_n_levels()
            module_prec = int(np.ceil(np.log2(module_levels)))
            prec_dict[n.name + '$'] = module_levels

            if isinstance(n.module, tuple(_BB_LINOPS)) and self.net is not None:
                if n.module.gate_ctrl.linked_layer is not None:
                    # get the bitwidth of the activation preceding the current
                    # linear layer
                    linked_act = [n_.module for n_ in self.nodes_list if self.fix_node_name(n_.name) == n.module.gate_ctrl.linked_layer][0]
                    n_levels = linked_act.get_n_levels() if isinstance(linked_act, BBAct) else linked_act.n_levels
                    in_prec = int(np.ceil(np.log2(linked_act.get_n_levels())))
                else:
                    #if a linear layer has no "linked_layer" registered, it
                    #should be the input layer.
                    in_prec = self.input_bitwidth
                n = LightweightNode(self.fix_node_name(n.name), n.module)
                fx_node = fx_node_of_ql_node(self.net, n)
                tot_bops += fx_node.meta['macs'] * in_prec * module_prec
                bops_8b += fx_node.meta['macs'] * 64

        bop_ratio = tot_bops/bops_8b
        out_dict = {
            'total_bops' : tot_bops,
            '8b_bops' : bops_8b,
            'bop_ratio': bop_ratio,
            'eff_bw': 8 * np.sqrt(bop_ratio),
            'layer_levels' : prec_dict}

        with open(self.export_file, 'w') as outfile:
            json.dump(out_dict, outfile, indent=4)
