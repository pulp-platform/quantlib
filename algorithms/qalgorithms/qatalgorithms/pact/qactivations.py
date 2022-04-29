import torch
import torch.nn as nn

from .qmodules import _PACTModule, _PACTActivation
from ...utils import ModuleMapping
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from quantlib.algorithms.qmodules import QIdentity, QReLU, QReLU6, QLeakyReLU


class PACTIdentity(_PACTActivation, QIdentity):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec)

        _PACTActivation.__init__(self)

    def _register_qop(self):
        super(PACTIdentity, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTIdentity, self).call_qop(x)


class PACTReLU(_PACTActivation, QReLU):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 inplace:                  bool = False):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          inplace=inplace)

        _PACTActivation.__init__(self)

    def _register_qop(self):
        super(PACTReLU, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTReLU, self).call_qop(x)


class PACTReLU6(_PACTActivation, QReLU6):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 inplace:                  bool = False):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          inplace=inplace)

        _PACTActivation.__init__(self)

    def _register_qop(self):
        super(PACTReLU6, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTReLU6, self).call_qop(x)


class PACTLeakyReLU(_PACTActivation, QLeakyReLU):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 negative_slope:           float = 1e-2,
                 inplace:                  bool = False):

        super(_PACTModule, self).__init__(qrangespec,
                                          qgranularityspec,
                                          qhparamsinitstrategyspec,
                                          negative_slope=negative_slope,
                                          inplace=inplace)

        _PACTActivation.__init__(self)

    def _register_qop(self):
        super(PACTLeakyReLU, self).register_qop()

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super(PACTLeakyReLU, self).call_qop(x)


NNMODULE_TO_PACTACTIVATION: ModuleMapping = {
    nn.Identity:  PACTIdentity,
    nn.ReLU:      PACTReLU,
    nn.ReLU6:     PACTReLU6,
    nn.LeakyReLU: PACTLeakyReLU,
}
