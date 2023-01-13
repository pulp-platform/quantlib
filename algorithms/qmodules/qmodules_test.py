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
import unittest
import torch
import torch.nn as nn
from typing import Tuple, Union

from .qmodules import QReLU
from .qmodules import QConv2d
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType


def _fake_quantise(x:       torch.Tensor,
                   clip_lo: torch.Tensor,
                   clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) -> torch.Tensor:

    x = torch.clip(x, clip_lo, clip_hi + scale / 4)
    x = x - clip_lo
    x = x / (step * scale)
    x = torch.floor(x)
    x = x * (step * scale)

    return x


class MockUpQReLU(QReLU):  # no real `torch.autograd.Function` object is registered

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 inplace:                  bool = False):

        super().__init__(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         inplace=inplace)

    def _register_qop(self):
        pass

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return _fake_quantise(x, self.clip_lo, self.clip_hi, self.step, self.scale)


class MockUpQConv2d(QConv2d):

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
                 bias:                     bool = False):

        super().__init__(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias=bias)

    def _register_qop(self):
        pass

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return _fake_quantise(x, self.clip_lo, self.clip_hi, self.step, self.scale)


_FEATURES_SHAPE           = (1, 8, 200, 200)
_BROADCAST_FEATURES_SHAPE = (1,)
_RELU_MODULE = nn.ReLU()

_IN_CHANNELS  = 8
_OUT_CHANNELS = 16
_KERNEL_SIZE  = 3
_HAS_BIAS     = False
_WEIGHTS_SHAPE           = (16, 8, 3, 3)
_BROADCAST_WEIGHTS_SHAPE = (16, 1, 1, 1)
_CONV2D_MODULE = nn.Conv2d(in_channels=_IN_CHANNELS, out_channels=_OUT_CHANNELS, kernel_size=_KERNEL_SIZE, bias=_HAS_BIAS)

_LOOP_LENGTH = 100


class QModulesTest(unittest.TestCase):

    @staticmethod
    def _check_broadcastability(reference: Union[torch.Size, Tuple[int, ...]], muqm: Union[MockUpQReLU, MockUpQConv2d]) -> bool:

        def _check_shape(reference: Union[torch.Size, Tuple[int, ...]], to_be_tested: Union[torch.Size, Tuple[int, ...]]) -> bool:
            return bool(to_be_tested == reference)

        cond_zero     = _check_shape(reference, muqm.zero.shape)
        cond_n_levels = _check_shape(reference, muqm.n_levels.shape)
        cond_step     = _check_shape(reference, muqm.step.shape)
        cond_scale    = _check_shape(reference, muqm.scale.shape)

        return cond_zero & cond_n_levels & cond_step & cond_scale

    @staticmethod
    def _check_is_unquantised_zero_and_scale(muqm: Union[MockUpQReLU, MockUpQConv2d]) -> bool:
        cond_zero  = bool(torch.all(torch.isnan(muqm.zero)))
        cond_scale = bool(torch.all(torch.isnan(muqm.scale)))
        cond_pin   = bool(muqm._pin_offset)
        cond_flag  = bool(muqm._is_quantised)
        return cond_zero & cond_scale & (not cond_pin) & (not cond_flag)

    @staticmethod
    def _check_is_unquantised_only_scale(muqm: Union[MockUpQReLU, MockUpQConv2d]) -> bool:
        cond_zero  = bool(torch.any(torch.isnan(muqm.zero)))
        cond_scale = bool(torch.all(torch.isnan(muqm.scale)))
        cond_pin   = bool(muqm._pin_offset)
        cond_flag  = bool(muqm._is_quantised)
        return (not cond_zero) & cond_scale & cond_pin & (not cond_flag)

    @staticmethod
    def _check_is_quantised(muqm: Union[MockUpQReLU, MockUpQConv2d]) -> bool:
        cond_zero  = bool(torch.any(torch.isnan(muqm.zero)))
        cond_scale = bool(torch.any(torch.isnan(muqm.scale)))
        # cond_pin   = bool(muqm._pin_offset)  # is irrelevant
        cond_flag  = bool(muqm._is_quantised)
        return (not cond_zero) & (not cond_scale) & cond_flag

    @staticmethod
    def _check_integerisation(reversedfq: torch.Tensor, tq: torch.Tensor) -> bool:
        absdiff = torch.abs(reversedfq - tq)
        cond = bool(torch.all((0.0 <= absdiff) & (absdiff <= 1.0)))  # I know that the absolute value has non-negative codomain, but I state the condition explicitly for readability
        return cond

    def test__qactivation_without_warmup(self):

        # error: unsupported granularity
        qrangespec = {'n_levels': 3}
        qgranularityspec = 'per-outchannel_weights'
        qhparamsinitstrategyspec = 'const'
        self.assertRaises(ValueError, lambda: MockUpQReLU(qrangespec, qgranularityspec, qhparamsinitstrategyspec))

        # cold strategy (const); unspecified offset/zero-point
        # create object
        qrangespec = {'n_levels': 3}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'const'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_zero_and_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # finalise quantiser parametrisation
        muqr.init_qhparams()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_quantised(muqr))
        self.assertFalse(muqr._is_observing)
        # fake-quantise input
        x = torch.randn(_FEATURES_SHAPE)
        fqy = muqr(x)
        tqy = fqy / (muqr.step * muqr.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqy, torch.floor(tqy)))

        # cold strategy (const); specified offset/zero-point
        # create object
        qrangespec = {'bitwidth': 8, 'signed': True}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'const'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # finalise quantiser parametrisation
        muqr.init_qhparams()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_quantised(muqr))
        self.assertFalse(muqr._is_observing)
        # fake-quantise input
        x = torch.randn(_FEATURES_SHAPE)
        fqy = muqr(x)
        tqy = fqy / (muqr.step * muqr.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqy, torch.floor(tqy)))

        # error: warmed-up strategy (minmax); unspecified offset/zero-point
        # create object
        qrangespec = {'n_levels': 3}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'minmax'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_zero_and_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # finalise quantiser parametrisation
        self.assertRaises(RuntimeError, lambda: muqr.init_qhparams())

        # error: warmed-up strategy (minmax); specified offset/zero-point
        # create object
        qrangespec = {'bitwidth': 8, 'signed': True}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'minmax'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # finalise quantiser parametrisation
        self.assertRaises(RuntimeError, lambda: muqr.init_qhparams())

    def test__qactivation_with_warmup(self):

        # cold strategy (const)
        # create object
        qrangespec = {'bitwidth': 8, 'signed': False}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'const'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # warm-up observer and finalise quantiser parametrisation
        muqr.start_observing()
        for i in range(0, _LOOP_LENGTH):
            self.assertTrue(muqr._is_observing)
            x = torch.randn(_FEATURES_SHAPE)
            _ = muqr(x)
        muqr.stop_observing()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_quantised(muqr))
        self.assertFalse(muqr._is_observing)
        # fake-quantise input
        x = torch.randn(_FEATURES_SHAPE)
        fqy = muqr(x)
        tqy = fqy / (muqr.step * muqr.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqy, torch.floor(tqy)))

        # warmed-up strategy (minmax)
        # create object
        qrangespec = {'bitwidth': 8, 'signed': False}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'minmax'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # warm-up observer and finalise quantiser parametrisation
        muqr.start_observing()
        for i in range(0, _LOOP_LENGTH):
            self.assertTrue(muqr._is_observing)
            x = torch.randn(_FEATURES_SHAPE)
            _ = muqr(x)
        muqr.stop_observing()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_quantised(muqr))
        self.assertFalse(muqr._is_observing)
        # fake-quantise input
        x = torch.randn(_FEATURES_SHAPE)
        fqy = muqr(x)
        tqy = fqy / (muqr.step * muqr.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqy, torch.floor(tqy)))

        # warmed-up strategy (meanstd)
        # create object
        qrangespec = {'bitwidth': 8, 'signed': False}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'meanstd'
        muqr = MockUpQReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqr))
        self.assertFalse(muqr._is_observing)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqr(x) == _RELU_MODULE(x)))
        self.assertTrue(muqr.inplace == _RELU_MODULE.inplace)
        # warm-up observer and finalise quantiser parametrisation
        muqr.start_observing()
        for i in range(0, _LOOP_LENGTH):
            self.assertTrue(muqr._is_observing)
            x = torch.randn(_FEATURES_SHAPE)
            _ = muqr(x)
        muqr.stop_observing()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqr))
        self.assertTrue(QModulesTest._check_is_quantised(muqr))
        self.assertFalse(muqr._is_observing)
        # fake-quantise input
        x = torch.randn(_FEATURES_SHAPE)
        fqy = muqr(x)
        tqy = fqy / (muqr.step * muqr.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqy, torch.floor(tqy)))

    def test__qlinear(self):

        # cold strategy (const); unspecified offset/zero-point, per-array granularity
        # create object
        qrangespec = {'n_levels': 3}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'const'
        muqc2d = MockUpQConv2d.from_fp_module(_CONV2D_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqc2d))
        self.assertTrue(QModulesTest._check_is_unquantised_zero_and_scale(muqc2d))
        self.assertFalse(muqc2d._is_observing)
        # does it have Conv2d behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqc2d(x) == _CONV2D_MODULE(x)))
        # finalise quantiser parametrisation
        muqc2d.init_qhparams()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqc2d))
        self.assertTrue(QModulesTest._check_is_quantised(muqc2d))
        self.assertFalse(muqc2d._is_observing)
        # fake-quantise weights
        fqw = muqc2d.qweight
        tqw = fqw / (muqc2d.step * muqc2d.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqw, torch.floor(tqw)))

        # warmed-up strategy (minmax); specified offset/zero-point, per-array granularity
        # create object
        qrangespec = {'bitwidth': 8, 'signed': False}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'minmax'
        muqc2d = MockUpQConv2d.from_fp_module(_CONV2D_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqc2d))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqc2d))
        self.assertFalse(muqc2d._is_observing)
        # does it have Conv2d behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqc2d(x) == _CONV2D_MODULE(x)))
        # finalise quantiser parametrisation (weights are already available it their entirety, so we don't need to warm-up statistic counters)
        muqc2d.init_qhparams()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_FEATURES_SHAPE, muqc2d))
        self.assertTrue(QModulesTest._check_is_quantised(muqc2d))
        self.assertFalse(muqc2d._is_observing)
        # fake-quantise weights
        fqw = muqc2d.qweight
        tqw = fqw / (muqc2d.step * muqc2d.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqw, torch.floor(tqw)))

        # warmed-up strategy (meanstd); specified offset/zero-point, per-outchannel granularity
        # create object
        qrangespec = {'bitwidth': 8, 'signed': False}
        qgranularityspec = 'per-outchannel_weights'
        qhparamsinitstrategyspec = 'meanstd'
        muqc2d = MockUpQConv2d.from_fp_module(_CONV2D_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_WEIGHTS_SHAPE, muqc2d))
        self.assertTrue(QModulesTest._check_is_unquantised_only_scale(muqc2d))
        self.assertFalse(muqc2d._is_observing)
        # does it have Conv2d behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(muqc2d(x) == _CONV2D_MODULE(x)))
        # warm-up observer and finalise quantiser parametrisation
        muqc2d.start_observing()
        for i in range(0, _LOOP_LENGTH):
            self.assertTrue(muqc2d._is_observing)
            x = torch.randn(_FEATURES_SHAPE)
            _ = muqc2d(x)
        muqc2d.stop_observing()
        self.assertTrue(QModulesTest._check_broadcastability(_BROADCAST_WEIGHTS_SHAPE, muqc2d))
        self.assertTrue(QModulesTest._check_is_quantised(muqc2d))
        self.assertFalse(muqc2d._is_observing)
        # fake-quantise weights
        fqw = muqc2d.qweight
        tqw = fqw / (muqc2d.step * muqc2d.scale)
        self.assertTrue(QModulesTest._check_integerisation(tqw, torch.floor(tqw)))
