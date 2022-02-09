import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .qmodules import _QModule, _QLinear


class QLinear(_QLinear, nn.Linear):

    def __init__(self,
                 qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):

        super(_QModule, self).__init__(in_features=in_features,
                                       out_features=out_features,
                                       bias=bias)
        _QLinear.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)


class QConv1d(_QLinear, nn.Conv1d):

    def __init__(self,
                 qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = 1,
                 padding: str = 0,
                 dilation: Tuple[int, ...] = 1,
                 groups: int = 1,
                 bias: bool = True):

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

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QConv2d(_QLinear, nn.Conv2d):

    def __init__(self,
                 qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = 1,
                 padding: str = 0,
                 dilation: Tuple[int, ...] = 1,
                 groups: int = 1,
                 bias: bool = True):

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

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QConv3d(_QLinear, nn.Conv3d):

    def __init__(self,
                 qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = 1,
                 padding: str = 0,
                 dilation: Tuple[int, ...] = 1,
                 groups: int = 1,
                 bias: bool = True):

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

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight

        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
