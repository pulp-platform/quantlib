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

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .qmodules import _QModule, _QLinear
from ...qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType


SUPPORTED_LINEAR_FPMODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)


class QLinear(_QLinear, nn.Linear):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_features:              int,
                 out_features:             int,
                 bias:                     bool = True):

        super(_QModule, self).__init__(in_features=in_features,
                                       out_features=out_features,
                                       bias=bias)
        _QLinear.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Linear,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QLinear:
        """Special constructor to build ``QLinear``s from FP ``Linear``s."""

        qlinear = cls(qrangespec,
                      qgranularityspec,
                      qhparamsinitstrategyspec,
                      in_features=fpm.in_features,
                      out_features=fpm.out_features,
                      bias=(fpm.bias is not None))

        # copy parameters over
        qlinear.weight.data.copy_(fpm.weight.data)
        if fpm.bias is not None:
            qlinear.bias.data.copy_(fpm.bias.data)

        return qlinear


class QConv1d(_QLinear, nn.Conv1d):

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

        super(_QModule, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias)

        _QLinear.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Conv1d,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QConv1d:
        """Special constructor to build ``QConv1d``s from FP ``Conv1d``s."""

        qconv1d = cls(qrangespec,
                      qgranularityspec,
                      qhparamsinitstrategyspec,
                      in_channels=fpm.in_channels,
                      out_channels=fpm.out_channels,
                      kernel_size=fpm.kernel_size,
                      stride=fpm.stride,
                      padding=fpm.padding,
                      dilation=fpm.dilation,
                      groups=fpm.groups,
                      bias=(fpm.bias is not None))

        # copy parameters over
        qconv1d.weight.data.copy_(fpm.weight.data)
        if fpm.bias is not None:
            qconv1d.bias.data.copy_(fpm.bias.data)

        return qconv1d


class QConv2d(_QLinear, nn.Conv2d):

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

        super(_QModule, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias)

        _QLinear.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Conv2d,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QConv2d:
        """Special constructor to build ``QConv2d``s from FP ``Conv2d``s."""

        qconv2d = cls(qrangespec,
                      qgranularityspec,
                      qhparamsinitstrategyspec,
                      in_channels=fpm.in_channels,
                      out_channels=fpm.out_channels,
                      kernel_size=fpm.kernel_size,
                      stride=fpm.stride,
                      padding=fpm.padding,
                      dilation=fpm.dilation,
                      groups=fpm.groups,
                      bias=(fpm.bias is not None))

        # copy parameters over
        qconv2d.weight.data.copy_(fpm.weight.data)
        if fpm.bias is not None:
            qconv2d.bias.data.copy_(fpm.bias.data)

        return qconv2d


class QConv3d(_QLinear, nn.Conv3d):

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

        super(_QModule, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias)

        _QLinear.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Conv3d,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QConv3d:
        """Special constructor to build ``QConv3d``s from FP ``Conv3d``s."""

        qconv3d = cls(qrangespec,
                      qgranularityspec,
                      qhparamsinitstrategyspec,
                      in_channels=fpm.in_channels,
                      out_channels=fpm.out_channels,
                      kernel_size=fpm.kernel_size,
                      stride=fpm.stride,
                      padding=fpm.padding,
                      dilation=fpm.dilation,
                      groups=fpm.groups,
                      bias=(fpm.bias is not None))

        # copy parameters over
        qconv3d.weight.data.copy_(fpm.weight.data)
        if fpm.bias is not None:
            qconv3d.bias.data.copy_(fpm.bias.data)

        return qconv3d


NNMODULE_TO_QMODULE = {
    nn.Linear: QLinear,
    nn.Conv1d: QConv1d,
    nn.Conv2d: QConv2d,
    nn.Conv3d: QConv3d,
}
