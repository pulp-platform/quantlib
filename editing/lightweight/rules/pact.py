#
# pact.py
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
from functools import partial
from copy import deepcopy

from torch import nn

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.generic import CausalConv1d
from .rules import LightweightRule
from .filters import Filter

def replace_pact_conv_linear(module : Union[nn.Conv1d, nn.Conv2d, nn.Linear, CausalConv1d],
                             **kwargs):
    if isinstance(module, CausalConv1d):
        return PACTCausalConv1d.from_causalconv1d(module, **kwargs)
    elif isinstance(module, nn.Conv1d):
        return PACTConv1d.from_conv1d(module, **kwargs)
    elif isinstance(module, nn.Conv2d):
        return PACTConv2d.from_conv2d(module, **kwargs)
    elif isinstance(module, nn.Linear):
        return PACTLinear.from_linear(module, **kwargs)
    else:
        raise TypeError(f"Incompatible module of type {module.__class__.__name__} passed to replace_pact_conv_linear!")

def replace_pact_act(module : nn.Module,
                     signed : bool = False,
                     **kwargs):
    if 'act_kind' not in kwargs.keys():
        if isinstance(module, nn.ReLU6):
            act_kind = 'relu6'
        elif isinstance(module, nn.LeakyReLU):
            act_kind = 'leaky_relu'
            if 'leaky' not in kwargs:
                kwargs['leaky'] = module.negative_slope
        else: # default activation is ReLU
            act_kind = 'relu'

        kwargs['act_kind'] = act_kind

    if signed:
        return PACTAsymmetricAct(**kwargs)
    else:
        return PACTUnsignedAct(**kwargs)

def replace_pact_hard_act(module : nn.Module,
                          hact_kwargs : dict,
                          quant_act_kwargs : dict):
    if isinstance(module, nn.Hardsigmoid):
        layers = [
            PACTHardsigmoid(**hact_kwargs),
            PACTUnsignedAct(**quant_act_kwargs)
        ]
    elif isinstance(module, nn.Hardswish):
        layers = [
            PACTHardswish(**hact_kwargs),
            PACTAsymmetricAct(**quant_act_kwargs)
        ]
    return nn.Sequential(*layers)


def pact_quant_hard_act(module : nn.Module,
                        quant_act_kwargs : dict):
    if isinstance(module, nn.Hardsigmoid):
        layers = [
            nn.Hardsigmoid(),
            PACTUnsignedAct(**quant_act_kwargs)
        ]
    elif isinstance(module, nn.Hardswish):
        layers = [
            nn.Hardswish(),
            PACTAsymmetricAct(**quant_act_kwargs)
        ]
    return nn.Sequential(*layers)

def quantize_pool_pact(module : nn.Module,
                       signed : bool,
                       **quant_act_kwargs):
    if signed:
        act = PACTAsymmetricAct(**quant_act_kwargs)
    else:
        act = PACTUnsignedAct(**quant_act_kwargs)

    return nn.Sequential(module, act)

class QuantizePoolingLayers(LightweightRule):
    def __init__(self,
                 filter_: Filter,
                 **kwargs):
        replacement_fun = partial(quantize_pool_pact, **kwargs)
        super(QuantizePoolingLayers, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


class ReplaceConvLinearPACTRule(LightweightRule):
    def __init__(self,
                 filter_: Filter,
                 **kwargs):
        replacement_fun = partial(replace_pact_conv_linear, **kwargs)
        super(ReplaceConvLinearPACTRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)

class ReplaceActPACTRule(LightweightRule):
    def __init__(self,
                 filter_: Filter,
                 signed : bool = False,
                 **kwargs):
        replacement_fun = partial(replace_pact_act, signed=signed, **kwargs)
        super(ReplaceActPACTRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


class ReplaceHardActPACTRule(LightweightRule):
    def __init__(self,
                 filter_: Filter,
                 hact_kwargs : dict = {},
                 quant_act_kwargs : dict = {},
                 use_pact_hact : bool = True):
        if use_pact_hact:
            replacement_fun = partial(replace_pact_hard_act, hact_kwargs=hact_kwargs, quant_act_kwargs=quant_act_kwargs)
        else:
            replacement_fun = partial(pact_quant_hard_act, quant_act_kwargs=quant_act_kwargs)
        super(ReplaceHardActPACTRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)
