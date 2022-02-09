import torch
import torch.nn as nn

from .qmodules import _QModule, _QActivation
from ...qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType


class QIdentity(_QActivation, nn.Identity):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        super(_QModule, self).__init__()

        _QActivation.__init__(self,
                              qrangespec,
                              qgranularityspec,
                              qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QReLU(_QActivation, nn.ReLU):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 inplace:                  bool = False):

        super(_QModule, self).__init__(inplace=inplace)

        _QActivation.__init__(self,
                              qrangespec,
                              qgranularityspec,
                              qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QReLU6(_QActivation, nn.ReLU6):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 inplace:                  bool = False):

        super(_QModule, self).__init__(inplace=inplace)

        _QActivation.__init__(self,
                              qrangespec,
                              qgranularityspec,
                              qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QLeakyReLU(_QActivation, nn.LeakyReLU):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 negative_slope:           float = 1e-2,
                 inplace:                  bool = False):

        super(_QModule, self).__init__(negative_slope=negative_slope,
                                       inplace=inplace)

        _QActivation.__init__(self,
                              qrangespec,
                              qgranularityspec,
                              qhparamsinitstrategyspec)

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
