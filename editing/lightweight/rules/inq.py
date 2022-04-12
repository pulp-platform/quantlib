# 
# inq.py
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

import torch
import torch.nn as nn
from functools import partial

from .rules import LightweightRule
import quantlib.algorithms as qa

from typing import Union

from .filters import Filter


def replace_conv2d_inqconv2d(module:            torch.nn.Module,
                             num_levels:        int,
                             quant_init_method: Union[str, None],
                             quant_strategy:    str) -> torch.nn.Module:

    assert type(module) == nn.Conv2d

    return qa.inq.INQConv2d(in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=True if module.bias is not None else False,
                            num_levels=num_levels,
                            quant_init_method=quant_init_method,
                            quant_strategy=quant_strategy)


class ReplaceConv2dINQConv2dRule(LightweightRule):

    def __init__(self,
                 filter_:           Filter,
                 num_levels:        int,
                 quant_init_method: Union[str, None] = None,
                 quant_strategy:    str = 'magnitude'):

        replacement_fun = partial(replace_conv2d_inqconv2d, num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)  # currying returns a function with the correct signature ``Callable[[torch.nn.Module], torch.nn.Module]``
        super(ReplaceConv2dINQConv2dRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)

