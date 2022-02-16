from __future__ import annotations
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

    @classmethod
    def from_fp_module(cls,
                       fpm: nn.Identity,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QIdentity:
        """Special constructor to build ``QIdentity``s from FP ``Identity``s."""
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

    @classmethod
    def from_fp_module(cls,
                       fpm: nn.ReLU,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QReLU:
        """Special constructor to build ``QReLU``s from FP ``ReLU``s."""
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

    @classmethod
    def from_fp_module(cls,
                       fpm: nn.ReLU6,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QReLU6:
        """Special constructor to build ``QReLU6``s from FP ``ReLU6``s."""
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

    @classmethod
    def from_fp_module(cls,
                       fpm: nn.LeakyReLU,
                       qrangespec: QRangeSpecType,
                       qgranularityspec: QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                       **kwargs) -> QLeakyReLU:
        """Special constructor to build ``QLeakyReLU``s from FP ``LeakyReLU``s."""
        raise NotImplementedError


NNMODULE_TO_QMODULE = {
    nn.Identity:  QIdentity,
    nn.ReLU:      QReLU,
    nn.ReLU6:     QReLU6,
    nn.LeakyReLU: QLeakyReLU,
}
