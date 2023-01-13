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

from .qmodules import _QModule, _QActivation
from ...qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType


SUPPORTED_ACTIVATION_FPMODULES = (
    nn.Identity,
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
)


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
                       fpm:                      nn.Identity,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QIdentity:
        """Special constructor to build ``QIdentity``s from FP ``Identity``s."""

        qidentity = cls(qrangespec,
                        qgranularityspec,
                        qhparamsinitstrategyspec)

        return qidentity


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
                       fpm:                      nn.ReLU,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QReLU:
        """Special constructor to build ``QReLU``s from FP ``ReLU``s."""

        qrelu = cls(qrangespec,
                    qgranularityspec,
                    qhparamsinitstrategyspec,
                    inplace=fpm.inplace)

        return qrelu


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
                       fpm:                      nn.ReLU6,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QReLU6:
        """Special constructor to build ``QReLU6``s from FP ``ReLU6``s."""

        qrelu6 = cls(qrangespec,
                     qgranularityspec,
                     qhparamsinitstrategyspec,
                     inplace=fpm.inplace)

        return qrelu6


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
                       fpm:                      nn.LeakyReLU,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QLeakyReLU:
        """Special constructor to build ``QLeakyReLU``s from FP ``LeakyReLU``s."""

        qleakyrelu = cls(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         negative_slope=fpm.negative_slope,
                         inplace=fpm.inplace)

        return qleakyrelu


NNMODULE_TO_QMODULE = {
    nn.Identity:  QIdentity,
    nn.ReLU:      QReLU,
    nn.ReLU6:     QReLU6,
    nn.LeakyReLU: QLeakyReLU,
}
