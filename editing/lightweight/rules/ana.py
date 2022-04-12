# 
# ana.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

import torch.nn as nn
from functools import partial

from .rules import LightweightRule
import quantlib.algorithms as qa

from .filters import Filter


def replace_relu_anaact(module:         nn.Module,
                        quantizer_spec: dict,
                        noise_type:     str,
                        strategy:       str):

    assert isinstance(module, nn.ReLU)
    return qa.ana.ANAActivation(quantizer_spec=quantizer_spec, noise_type=noise_type, strategy=strategy)


def replace_linear_analinear(module:         nn.Module,
                             quantizer_spec: dict,
                             noise_type:     str,
                             strategy:       str) -> nn.Module:

    assert isinstance(module, nn.Linear)

    return qa.ana.ANALinear(quantizer_spec=quantizer_spec,
                            noise_type=noise_type,
                            strategy=strategy,
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=True if module.bias is not None else False)


def replace_conv2d_anaconv2d(module:         nn.Module,
                             quantizer_spec: dict,
                             noise_type:     str,
                             strategy:       str) -> nn.Module:

    assert isinstance(module, nn.Conv2d)

    return qa.ana.ANAConv2d(quantizer_spec=quantizer_spec,
                            noise_type=noise_type,
                            strategy=strategy,
                            in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias)


class ReplaceReLUANAActivationRule(LightweightRule):

    def __init__(self,
                 filter_:        Filter,
                 quantizer_spec: dict,
                 noise_type:     str,
                 strategy:       str):

        replacement_fun = partial(replace_relu_anaact, quantizer_spec=quantizer_spec, noise_type=noise_type, strategy=strategy)
        super(ReplaceReLUANAActivationRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


class ReplaceLinearANALinearRule(LightweightRule):

    def __init__(self,
                 filter_:        Filter,
                 quantizer_spec: dict,
                 noise_type:     str,
                 strategy:       str):

        replacement_fun = partial(replace_linear_analinear, quantizer_spec=quantizer_spec, noise_type=noise_type, strategy=strategy)
        super(ReplaceLinearANALinearRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


class ReplaceConv2dANAConv2dRule(LightweightRule):

    def __init__(self,
                 filter_:        Filter,
                 quantizer_spec: dict,
                 noise_type:     str,
                 strategy:       str):

        replacement_fun = partial(replace_conv2d_anaconv2d, quantizer_spec=quantizer_spec, noise_type=noise_type, strategy=strategy)
        super(ReplaceConv2dANAConv2dRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)

