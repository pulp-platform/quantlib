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

import torch
import torch.nn as nn
from typing import Tuple

from .qmodules import _PACTModule, _PACTLinear
from quantlib.algorithms.qalgorithms import ModuleMapping
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from quantlib.algorithms.qmodules import QLinear, QConv1d, QConv2d, QConv3d


class PACTLinear(_PACTLinear, QLinear):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_features:              int,
                 out_features:             int,
                 bias:                     bool = True):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)

        _PACTLinear.__init__(self)

    def _register_qop(self):
        super(PACTLinear, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTLinear, self).call_qop(x)


class PACTConv1d(_PACTLinear, QConv1d):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  str = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 bias:                     bool = True):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        _PACTLinear.__init__(self)

    def _register_qop(self):
        super(PACTConv1d, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTConv1d, self).call_qop(x)


class PACTConv2d(_PACTLinear, QConv2d):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  str = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 bias:                     bool = True):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        _PACTLinear.__init__(self)

    def _register_qop(self):
        super(PACTConv2d, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTConv2d, self).call_qop(x)


class PACTConv3d(_PACTLinear, QConv3d):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  str = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 bias:                     bool = True):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        _PACTLinear.__init__(self)

    def _register_qop(self):
        super(PACTConv3d, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTConv3d, self).call_qop(x)


NNMODULE_TO_PACTLINEAR = ModuleMapping([
    (nn.Linear, PACTLinear),
    (nn.Conv1d, PACTConv1d),
    (nn.Conv2d, PACTConv2d),
    (nn.Conv3d, PACTConv3d),
])
