from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .qmodules import _QModule, _QLinear
from ...qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType


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
                       fpm: nn.Linear,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QLinear:
        """Special constructor to build ``QLinear``s from FP ``Linear``s."""
        raise NotImplementedError


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
                       fpm: nn.Conv1d,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QConv1d:
        """Special constructor to build ``QConv1d``s from FP ``Conv1d``s."""
        raise NotImplementedError


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
                       fpm: nn.Conv2d,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QConv2d:
        """Special constructor to build ``QConv2d``s from FP ``Conv2d``s."""
        raise NotImplementedError


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
                       fpm: nn.Conv3d,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QConv3d:
        """Special constructor to build ``QConv3d``s from FP ``Conv3d``s."""
        raise NotImplementedError


NNMODULE_TO_QMODULE = {
    nn.Linear: QLinear,
    nn.Conv1d: QConv1d,
    nn.Conv2d: QConv2d,
    nn.Conv3d: QConv3d,
}
