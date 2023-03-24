#
# pact_controllers.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
#
# Copyright (c) 2020-2021 ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Union

import torch
from torch import nn
import numpy as np

from quantlib.editing.lightweight import LightweightGraph
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, SubTypeFilter, TypeFilter
from ..controller import Controller

from .pact_ops import *
from .util import assert_param_valid, almost_symm_quant, mse_bounds

import copy

import math

__all__ = [
    'PACTEpsController',
    'PACTActController',
    'PACTLinearController',
    'PACTIntegerModulesController',
    'PACTDynamicPrecController'
]


# Coefficients generated with this code:
# https://colab.research.google.com/drive/1IH8TwCfkvcMkHx4tHldMNgsvBIgCpTeM?usp=sharing
# first entry is c1, second is c2 with alpha_w^* = c1 * sqrt(Ew2) - c2 * Ew1
_sawb_symm_lut = {
      3: [ 2.58072,  1.68371],
      4: [ 3.22824,  2.19780],
      7: [ 6.99146,  6.34798],
      8: [ 7.57111,  6.96333],
     15: [12.04287, 12.07040],
     16: [12.16637, 12.19222],
     31: [14.71474, 15.13775],
     32: [15.18684, 15.72128],
     63: [20.78714, 22.60234],
     64: [20.54678, 22.29264],
    127: [27.15792, 30.50641],
    128: [27.75226, 31.25157],
    255: [35.48044, 40.89530],
    256: [34.97231, 40.26019],
}

_sawb_asymm_lut = {
      3: [ 2.58072,  1.68371],
      4: [ 4.74932,  3.32860],
      7: [ 6.99146,  6.34798],
      8: [ 8.26900,  7.36509],
     15: [12.04287, 12.07040],
     16: [12.06485, 11.86833],
     31: [14.71474, 15.13775],
     32: [15.47646, 15.98616],
     63: [20.78714, 22.60234],
     64: [21.96366, 24.03408],
    127: [27.15792, 30.50641],
    128: [28.26410, 31.87972],
    255: [35.48044, 40.89530],
    256: [36.15292, 41.73554],
}

class PACTEpsController(Controller):
    def __init__(self, fx_model, modules, schedule, tracer, eps_pass, verbose = False):
        # If `fx_model` is an instance of DataParallel, we have to strip 'module'
        # from node names
        if isinstance(fx_model, nn.DataParallel):
            self.model = fx_model.module
        else:
            self.model = fx_model

        self.modules = modules
        self.schedule = {int(k):v.lower() if isinstance(v, str) else [val.lower() for val in v] for k,v in schedule.items()}
        self.verbose = verbose

        self.eps_pass = eps_pass
        # Choose a tracer that doesn't have PACTWrapModule!
        self.tracer = tracer

    def step_pre_training_batch(self, *args, **kwargs):
        fx_graph = self.tracer.trace(self.model)
        fx_model = torch.fx.GraphModule(self.tracer.root, fx_graph, self.tracer.root.__class__.__name__)
        fx_model = self.eps_pass.apply(fx_model)
        nm = dict(fx_model.named_modules())

        for node in fx_graph.nodes:
            if node.op == 'call_module' and nm[node.target] in self.modules:
                arg_eps_ins = node.meta['quant'].eps_in[0]
                nm[node.target].set_eps_in(arg_eps_ins)

    def step_pre_training_epoch(self, epoch, *args, **kwargs):
        if epoch in self.schedule.keys():
            cur_cmds = self.schedule[epoch]
            if not isinstance(cur_cmds, list):
                cur_cmds = [cur_cmds]
            for cmd in cur_cmds:
                self.log("Epoch {} - running command {}".format(epoch, cmd))
                if cmd == 'verbose_on':
                    self.verbose = True
                    self.log("Verbose mode enabled!")
                elif cmd == 'verbose_off':
                    self.verbose = False
                elif cmd == 'start':
                    for m in self.modules:
                        m.started |= True
                    self.log("Started epsilon propagation!")
                elif cmd == 'start_no_init':
                    m.started |= True
                    self.log("Started epsilon propagation without initialization!")
                elif cmd == 'stop':
                    for m in self.modules:
                        m.started &= False
                    self.log("Stopped epsilon propagation!")

    def step_pre_validation_epoch(self, epoch, *args, **kwargs):
        self.step_pre_training_batch(self, *args, **kwargs)
        if epoch in self.schedule.keys():
            cur_cmds = self.schedule[epoch]
            if not isinstance(cur_cmds, list):
                cur_cmds = [cur_cmds]
            for cmd in cur_cmds:
                if cmd == 'lock':
                    for m in self.modules:
                        m.locked |= True
                    self.log("Locked epsilon!")

    def state_dict(self):
        return {'verbose':self.verbose}

    def load_state_dict(self, state_dict : dict):
        try:
            self.verbose = state_dict['verbose']
        except KeyError:
            vo = self.verbose
            self.verbose = True
            self.log("Got a bad state_dict - ignoring!")
            self.verbose = vo

    def log(self, msg : str):
        if self.verbose:
            print("[PACTEpsController]      ", msg)

    @staticmethod
    def get_modules(net : nn.Module):
        net_nodes = LightweightGraph.build_nodes_list(net, leaf_types=(PACTITAMax, PACTITAPartialMax))
        filter_eps = SubTypeFilter(_PACTEps)
        return [n.module for n in filter_eps(net_nodes)]


class PACTActController(Controller):
    """
    Controller for PACT activation classes:
      - PACTUnsignedAct
      - PACTAsymmetricAct
    The controller takes a schedule telling it to enable/disable quantization
    at given epochs. Starting quantization entails setting the `quantize` flags
    to `True` and setting the initial clipping values based on the `init_clip` value.
    """
    def __init__(self, modules : list, schedule : dict, verbose : bool = False, init_clip_lo : float = -1., init_clip_hi : float = 1.):
        assert all(isinstance(m, _PACTActivation) for m in modules), "Non-activation modules passed to PACTActController!"

        self.modules = modules
        self.schedule = {int(k): v.lower() if isinstance(v, str) else [val.lower() for val in v] for k,v in schedule.items()}
        self.verbose = verbose
        self.init_clip_lo = init_clip_lo
        self.init_clip_hi = init_clip_hi
        self.frozen = False

    def step_pre_training_batch(self, *args, **kwargs):
        for m in self.modules:
            if m.tqt:
                max_val = (2**m.log_t.data.clone().detach())
                if isinstance(m, PACTUnsignedAct):
                    m.clip_hi.data.copy_(max_val)
                else:
                    clip_lo, clip_hi = almost_symm_quant(max_val, m.n_levels)
                    m.clip_lo.data.copy_(clip_lo)
                    m.clip_hi.data.copy_(clip_hi)
            
    def step_pre_training_epoch(self, epoch: int, *args, **kwargs):
        """
        Executed before every training epoch. If the current epoch is in the schedule, perform the indicated action:
        - 'start': start quantization - set the clipping bound(s) according to the method set as `init_clip` and set
          `quantize` to True
        - 'freeze': set `requires_grad` to False for all clipping parameters so they are not updated anymore
        - 'thaw':   set `requires_grad` to True for all clipping parameters
        - 'stop':   set `quantize` to False and set `requires_grad` to False for all clipping parameters
        :param epoch: The current epoch
        """

        if epoch in self.schedule.keys():
            cur_cmds = self.schedule[epoch]
            if not isinstance(cur_cmds, list):
                cur_cmds = [cur_cmds]
            for cmd in cur_cmds:
                self.log("Epoch {} - running command {}".format(epoch, cmd))
                if cmd == 'verbose_on':
                    self.verbose = True
                    self.log("Verbose mode enabled!")
                elif cmd == 'verbose_off':
                    self.verbose = False

                elif cmd == 'ready':
                    for m in self.modules:
                        m.ready |= True
                        m.updateClipBounds()
                        m.histogram *= 0
                        m.truemax /= m.truemax
                        m.truemin /= m.truemin
                    self.log("Started activation quantization!")

                elif cmd == 'start':
                    for m in self.modules:
                        self.reset_clip_bounds(m, m.init_clip)
                        m.started |= True
                    self.log("Started activation quantization!")

                elif cmd == 'start_no_init':
                    for m in self.modules:
                        m.started |= True
                    self.log("Started activation quantization without initialization!")

                elif cmd == 'freeze':
                    for m in self.modules:
                        for p in m.clipping_params.values():
                            p.requires_grad = False
                            p.grad = None
                    self.frozen = True
                    self.log("Froze clipping parameters!")

                elif cmd == 'thaw':
                    # 'thaw' means enable learning for clipping parameters
                    for m in self.modules:
                        for k, p in m.clipping_params.items():
                            try:
                                symm = m.symm
                            except AttributeError:
                                symm = False
                            p.requires_grad = m.learn_clip and not (k=='high' and symm) and not (k in ['low', 'high'] and m.tqt)
                    self.frozen = False
                    self.log("Unfroze clipping parameters!")

                elif cmd == 'stop':
                    for m in self.modules:
                        m.started &= False
                    self.log("Stopped quantization!")

    def reset_clip_bounds(self, m : Union[PACTUnsignedAct, PACTAsymmetricAct], method : str = None):
        if not method:
            method = m.init_clip
        if method in ['max', 'mse']:
            max_val = m.max.data
            if isinstance(m, PACTAsymmetricAct) and m.symm:
                max_val = torch.maximum(max_val, -m.min)
                min_val, max_val = almost_symm_quant(max_val, m.n_levels)
            else:
                min_val = m.min.data
        elif method == 'const':
            for p in m.clipping_params.values():
                max_val = torch.ones_like(p.data) * self.init_clip_hi
                min_val = torch.ones_like(p.data) * self.init_clip_lo
                if isinstance(m, PACTAsymmetricAct) and  m.symm:
                    max_abs = torch.maximum(max_val, -min_val)
                    min_val, max_val = almost_symm_quant(max_abs, m.n_levels)

                break

        elif method == 'percentile':
            m.updateClipBounds()
            max_val = m.max.data
            try:
                if m.symm:
                    max_val = torch.maximum(max_val, -m.min)
                    min_val, max_val = almost_symm_quant(max_val, m.n_levels)
                else:
                    min_val = m.min.data
            except AttributeError:
                # we don't need 'min_val' if m does not have the 'symm' attribute because in that case
                # it's not an asymmetric activation
                pass

        elif method == 'klj':

            m.updateClipBounds()

            l1 = torch.nn.KLDivLoss()
            l2 = torch.nn.KLDivLoss()

            def klj_loss(x_hp, x_fq, l1, l2):
                return l1(x_hp, x_fq) + l2(x_fq, x_hp)
            def jensen_shannon_loss(x_hp, x_fq, l1, l2):
                return 0.5*(l1(x_hp, ((x_fq+x_hp)/2)) + l2(x_fq, ((x_fq+x_hp)/2)))
            def cust_loss(x_hp, x_fq, bincenter, l1, l2):
                return torch.sum((torch.abs(x_hp-x_fq) * torch.abs(bincenter))**2)#* torch.cos(torch.abs(bincenter)/torch.max(bincenter) * pi/2)

            class Resampler(nn.Module):
                def __init__(self,m):
                    super().__init__()
                    self.m = m

                def forward(self, histogram, prevEdges, n_levels, max):
                    leftIdx = torch.sum(torch.where(prevEdges < -max, 1, 0))
                    rightIdx = torch.sum(torch.where(prevEdges < max, 1, 0))
                    newHist = torch.zeros_like(histogram)
                    oldHist = histogram[leftIdx:rightIdx]
                    innerSamples = torch.nn.functional.interpolate(oldHist.reshape(1,1,-1),size=n_levels,mode='linear').reshape(-1)
                    innerSamples[0] = innerSamples[0] + torch.sum(histogram[:leftIdx])
                    innerSamples[-1] = innerSamples[-1] + torch.sum(histogram[rightIdx:])
                    innerSamples = torch.nn.functional.interpolate(innerSamples.reshape(1,1,-1),size=(rightIdx-leftIdx),mode='linear').reshape(-1)
                    newHist[leftIdx:rightIdx] = innerSamples
                    return newHist

            model = Resampler(m)

            hist = copy.deepcopy(m.histogram/torch.sum(m.histogram))
            prevEdges = copy.deepcopy(m.prevEdges)
            binCenters = torch.zeros_like(hist)
            for idx in range(len(binCenters)):
                binCenters[idx] = (prevEdges[idx] + prevEdges[idx+1])/2
            abs_scale = m.truemax / m.num_bins

            best = 0
            minLoss = 1

            max = copy.deepcopy(m.truemax)

            maxHat = max
            model.eval()
            for i in range(m.num_bins-1):
                newHist = model(hist, prevEdges, m.n_levels, maxHat)
                absLoss = cust_loss(hist, newHist, binCenters, l1, l2)
                maxHat = maxHat - abs_scale
                if absLoss < minLoss:
                    best = i
                    minLoss = absLoss.item()

            max_val = m.truemax - abs_scale*best
            if m.symm:
                min_val, max_val = almost_symm_quant(max_val, m.n_levels)
            else:
                min_val = m.min.data

        else: # method == 'std'
            max_val = m.running_mean.data + m.nb_std * torch.sqrt(m.running_var.data)
            if isinstance(m, PACTAsymmetricAct) and m.symm:
                min_val, max_val = almost_symm_quant(max_val, m.n_levels)
            else:
                min_val = m.running_mean.data - m.nb_std * torch.sqrt(m.running_var.data)


        for k, b in m.clipping_params.items():
            if k == 'high':
                b.data = max_val
            elif k == 'low':
                b.data = min_val

        if m.tqt:
            # initialize the log_t parameter correctly
            if isinstance(m, PACTUnsignedAct):
                log_t = torch.log2(m.clip_hi)
            else:
                log_t = torch.log2(-m.clip_lo)
            m.log_t.data.copy_(log_t.reshape(m.log_t.shape))


    def step_pre_validation_epoch(self, *args, **kwargs):
        self.step_pre_training_batch()

    def log(self, msg : str):
        if self.verbose:
            print("[PACTActController]      ", msg)

    def state_dict(self):
        return {'verbose':self.verbose, 'frozen':self.frozen}

    def load_state_dict(self, state_dict : dict):
        try:
            self.verbose = state_dict['verbose']
            if state_dict['frozen']:
                for m in self.modules:
                    for p in m.clipping_params.values():
                        p.requires_grad = False
                        p.grad = None
                self.frozen = True
                self.log("All modules frozen!")
        except KeyError:
            vo = self.verbose
            self.verbose = True
            self.log("Got a bad state_dict - ignoring!")
            self.verbose = vo

    @staticmethod
    def get_modules(net : nn.Module):
        net_nodes = LightweightGraph.build_nodes_list(net)
        filter_act = SubTypeFilter(_PACTActivation)
        return [n.module for n in filter_act(net_nodes)]



class PACTLinearController(Controller):
    """
    Controller for PACT Linear classes:
      - PACTConv2d
      - PACTConv1d
      - PACTCausalConv1d
      - PACTLinear
    """

    def __init__(self, modules : list, schedule : dict, verbose : bool = False, init_clip_lo : float = -1., init_clip_hi : float = 1., update_every : str = 'batch'):
        assert_param_valid(self, update_every, 'update_every', ['batch', 'epoch'])
        super(PACTLinearController, self).__init__()
        self.modules = modules
        self.schedule = {int(k):v.lower() if isinstance(v, str) else [val.lower() for val in v] for k,v in schedule.items()}
        self.verbose = verbose
        self.init_clip_lo = init_clip_lo
        self.init_clip_hi = init_clip_hi
        self.frozen = False
        self.update_every = update_every

    def step_pre_training_batch(self, *args, **kwargs):
        if self.update_every == 'batch':
            self.update_clip_params()
        else: # TQT modules always need to be updated every batch to ensure
            # clip_lo and clip_hi are 2**log_t
            for m in self.modules:
                if m.tqt and m.started and not m.frozen:
                    self.reset_clip_bounds(m, None, init=False)


    def update_clip_params(self):
        with torch.no_grad():
            for m in self.modules:
                if (not m.frozen) and m.started:
                    if not m.learn_clip or m.tqt:
                        self.reset_clip_bounds(m, m.init_clip, init=False)
                    # if 'learn_clip' is True and 'symm_wts' is also True, we learn the lower clip bound and set the upper
                    # one automatically with a function equivalent to 'almost_symm_quant'. This is performed in the
                    # conv/linear module itself to ensure proper gradient propagation (???).
                    # However, we also update the upper clipping bound for layers where 'symm_wts' and 'learn_clip'
                    # here so it reflects the value that is used in forward propagation.
                    elif m.learn_clip and m.symm_wts and not m.tqt:
                        # if we learn symmetric weight bounds, it can happen
                        # that the lower bound is pushed past 0. In this step,
                        # we make sure that the lower bound stays smaller or
                        # equal to zero.
                        if torch.any(m.clip_lo.data>0):
                            self.log("Found a clip_lo that was >0: {}".format(m.clip_lo.data))
                            self.log("Clamping to -0.01!")
                        m.clip_lo.data = torch.minimum(torch.zeros_like(m.clip_lo.data)-0.01, m.clip_lo.data)
                        _, max_val = almost_symm_quant(-m.clip_lo.data, m.n_levels)
                        m.clip_hi.data = max_val

    def step_pre_training_epoch(self, epoch : int, *args, **kwargs):
        # keep track of whether we already performed update_clip_params
        do_update = True
        if epoch in self.schedule.keys():
            cur_cmds = self.schedule[epoch]
            if not isinstance(cur_cmds, list):
                cur_cmds = [cur_cmds]
            for cmd in cur_cmds:
                self.log("Epoch {} - running command {}".format(epoch, cmd))
                if cmd == 'verbose_on':
                    self.verbose = True
                    self.log("Verbose mode enabled!")

                elif cmd == 'verbose_off':
                    self.verbose = False

                elif cmd == 'start':
                    for m in self.modules:
                        self.reset_clip_bounds(m, init=True)
                        m.started |= True

                    self.log("Started quantization!")
                    do_update = False

                elif cmd == 'start_no_init':
                    self.log("Started quantization!")
                    for m in self.modules:
                        m.started |= True
                    do_update = False

                elif cmd == 'freeze':
                    #NOTE: this does not work as intended when using stateful
                    #optimizers such as Adam. The parameters will not stay
                    #frozen because the optimizer still applies momentum...
                    for m in self.modules:
                        for b in m.clipping_params.values():
                            b.requires_grad = False
                            b.grad = None
                        m.frozen |= True
                    self.frozen = True
                    do_update = False

                elif cmd == 'thaw':
                    for m in self.modules:
                        for k, b in m.clipping_params.items():
                            # if symm_wts is True, the upper bound is not learned but inferred from the lower bound.
                            b.requires_grad = m.learn_clip and not (k=='high' and m.symm_wts) and not (k in ['low', 'high'] and m.tqt)
                        m.frozen &= False
                    self.frozen = False

                else:
                    assert cmd == 'stop', "Invalid PACTLinearController command at epoch {}: {}".format(epoch, cmd)
                    for m in self.modules:
                        m.started &= False
                    self.log("Stopped quantization!")
                    do_update = False

        if self.update_every == 'epoch' and do_update:
            self.update_clip_params()


    def step_pre_validation_epoch(self, epoch: int, *args, **kwargs):
        # always before validation, update the clipping parameters as is done before each batch, so the changes from
        # the last batch of an epoch are reflected in the clipping params.
        self.step_pre_training_batch()

    # resetting clip bounds is almost identical between the different convolutions
    def reset_clip_bounds(self, m: Union[PACTConv2d, PACTConv1d, PACTLinear, PACTCausalConv1d], method: str = None, init : bool = False):
        if method is None:
            method = m.init_clip
        w = m.weight.data
        if m.quantize == 'per_channel':
            reduce_dims = tuple(range(1, len(w.shape)))
        else:
            reduce_dims = tuple(range(len(w.shape)))

        if init or not m.tqt:
            if method == 'max':
                if m.symm_wts:
                    max_val = torch.amax(w.abs(), dim=reduce_dims)
                    # if symm_wts is true, we do "almost symmetric" quantization in the case of an even n_levels (to account for e.g. int8 range going from -128 to 127)
                    min_val, max_val = almost_symm_quant(max_val, m.n_levels)
                else:
                    max_val = torch.amax(w, dim=reduce_dims)
                    min_val = torch.amin(w, dim=reduce_dims)
            elif method == 'std':
                max_val = w.mean(dim=reduce_dims) + w.std(dim=reduce_dims) * m.nb_std
                if m.symm_wts:
                    # 'std' initialization is inherently symmetrical, so use the "almost_symmetrical" quantization anyway
                    min_mal, max_val = almost_symm_quant(max_val, m.n_levels)
                else:
                    min_val = w.mean(dim=reduce_dims) - w.std(dim=reduce_dims) * m.nb_std

            elif method == 'mse':
                min_val, max_val = mse_bounds(w, m.n_levels, True, m.quantize=='per_channel', True, m.mse_iters, m.symm_wts)

            elif method[0:4] == 'sawb':
                symm = method[5:] == 'symm'
                # mean absolute weights: E[|W|] - either channel-wise or over the whole tensor depending on reduce_dims
                e_w_abs = m.weight.data.abs().mean(dim=reduce_dims)
                e_w2 = torch.mean(m.weight.data**2, dim=reduce_dims)
                if symm:
                    coeffs = _sawb_symm_lut[m.n_levels]
                else:
                    coeffs = _sawb_asymm_lut[m.n_levels]
                alpha = coeffs[0] * e_w2.sqrt() - coeffs[1] * e_w_abs


                # calculate the min/max bounds as well for the cases where SAWB
                # produces a negative alpha
                max_val_mm = torch.amax(w.abs(), dim=reduce_dims)
                # if symm_wts is true, we do "almost symmetric" quantization in the case of an even n_levels (to account for e.g. int8 range going from -128 to 127)
                min_val_mm, max_val_mm = almost_symm_quant(max_val_mm, m.n_levels)

                if symm:
                    min_val = -alpha
                    max_val = alpha
                else:
                    min_val, max_val = almost_symm_quant(alpha, m.n_levels)

                # where alpha is negative, use min/max bounds
                min_val, max_val = torch.where(alpha < 0, min_val_mm, min_val), torch.where(alpha < 0, max_val_mm, max_val)

            else: # method == 'const'
                min_val = torch.ones_like(m.clip_lo.data) * self.init_clip_lo
                max_val = torch.ones_like(m.clip_hi.data) * self.init_clip_hi
        if m.tqt:
            if init:
                # we already found the initial min_val and max_val, so we just
                # initialize the log_t parameter
                m.log_t.data.copy_(m.expand_bounds(torch.log2(-min_val)))
            else:
                # log_t is being learned, so update clip_lo and clip_hi to
                # almost_symm_quant(2**log_t)
                min_val, max_val = almost_symm_quant(2**m.log_t.clone().detach(), m.n_levels)

        m.clip_hi.data = m.expand_bounds(max_val)
        m.clip_lo.data = m.expand_bounds(min_val)

    def log(self, msg : str):
        if self.verbose:
            print("[PACTLinearController]   ", msg)

    def state_dict(self):
        return {'verbose': self.verbose, 'frozen':self.frozen}

    def load_state_dict(self, state_dict : dict):
        try:
            self.verbose = state_dict['verbose']

            if state_dict['frozen']:
                for m in self.modules:
                    for p in m.clipping_params.values():
                        p.requires_grad = False
                        p.grad = None
                self.frozen = True
                self.log("All modules frozen!")
        except KeyError:
            vo = self.verbose
            self.verbose = True
            self.log("Got a bad state_dict - ignoring!")
            self.verbose = vo

    @staticmethod
    def get_modules(net : nn.Module):
        filter_all_linops = SubTypeFilter(_PACTLinOp)
        net_nodes = LightweightGraph.build_nodes_list(net)
        return [n.module for n in filter_all_linops(net_nodes)]


class PACTIntegerModulesController(Controller):
    # a very simple controller which keeps the epsilons of PACTIntegerXXX nodes
    # synchronized before every inference
    def __init__(self, modules):
        self.modules = modules

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def step_pre_training_epoch(self, *args, **kwargs):
        pass

    def step_pre_training_batch(self, *args, **kwargs):
        # this is the only thing we do here...
        for m in self.modules:
            m.reassign_epsilons()

    def step_pre_validation_epoch(self, *args, **kwargs):
        # we need to sync the epsilons also after the last batch of a training epoch
        self.step_pre_training_batch(self, *args, **kwargs)

    @staticmethod
    def get_modules(net : nn.Module):
        net_nodes_intmodules_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd, PACTIntegerConcat))
        filter_intmodules = TypeFilter(PACTIntegerAdd) | TypeFilter(PACTIntegerConcat)
        return [n.module for n in filter_intmodules(net_nodes_intmodules_intact)]

class PACTDynamicPrecController(Controller):

    def __init__(self, module_spec_list : list):
        # module_spec_list is a list of tuples with each entry taking the form:
        # (m : nn.Module, levels : list[int], select_levels_trn : callable, select_levels_val : callable)
        # m is a PACT op module with an n_levels member
        # levels is a list of permissible n_levels parameters
        # select_levels_trn is a callable which takes 2 parameters:
        #  - l : list   -- the `levels` list
        #  - e : epoch  -- the current epoch; this allows us to e.g. implement
        #                  some type of annealing
        #   it is called in step_pre_training_batch and should return ann
        #   element of `l` to be used for training the current batch. Most
        #   commonly, this will be an implementation of the uniform distribution.
        # select_levels_val is expected to be the same type as
        # select_levels_trn but is used for validation - most commonly this
        # will be a constant function.
        self.module_spec_list = module_spec_list
        self.epoch = 0

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def step_pre_training_epoch(self, e : int, *args, **kwargs):
        self.epoch = e

    def step_pre_training_batch(self, *args, **kwargs):
        # all we need to do here is call the selection function for each module
        # and set the module's `n_levels` parameter to the result
        for modules, l, sel, _  in self.module_spec_list:
            # if "modules" is a list, we can apply the same (random)
            # quantization policy to multiple modules
            if not isinstance(modules, list):
                modules = [modules]
            nl = sel(l, self.epoch)
            # to constrain low-p weights to be the MSBs of high-p weights, we
            # only use rounded weights if we are using the highest available
            # precision.

            for m in modules:
                m.n_levels = nl

    def step_pre_validation_epoch(self, e : int, *args, **kwargs):
        # same as pre_training_batch but with the select_levels_val function
        for modules, l, _, sel in self.module_spec_list:
            if not isinstance(modules, list):
                modules = [modules]
            nl = sel(l, self.epoch)
            for m in modules:
                m.n_levels = nl
