# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
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

import unittest
from collections import OrderedDict
import torch
import torch.nn as nn

from quantlib.editing.graphs.nn import EpsTunnel


_EPS = torch.Tensor([1.0])
_N_FEATURES = 1
_KERNEL_SIZE = 1
_HAS_BIAS = False  # It does not make sense to set `bias = True`, since in
                   # such a case the output of the linear `nn.Module` would
                   # not be fake-quantised, and an `EpsTunnel` would not be
                   # inserted after it. Thus, the corresponding pattern should
                   # never appear, and a test for such a case would be
                   # meaningless. This is different from the test suite of
                   # `LinearOpBNBiasFolder`, where the linear operation in the
                   # pattern can appear both in canonical (`bias = False`) and
                   # non-canonical form (`bias = True`).


class EpsConv2dEps(nn.Sequential):

    def __init__(self):

        modules = OrderedDict([
            ('eps_in',  EpsTunnel(eps=_EPS)),
            ('conv2d',  nn.Conv2d(in_channels=_N_FEATURES, out_channels=_N_FEATURES, kernel_size=_KERNEL_SIZE, bias=_HAS_BIAS)),
            ('eps_out', EpsTunnel(eps=_EPS)),
        ])

        super(EpsConv2dEps, self).__init__(modules)
