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
import torch
import torch.nn as nn

from .pact import PACTReLU, PACTConv2d


_FEATURES_SHAPE    = (1, 8, 200, 200)
_FEATURES_ONES     = torch.ones(_FEATURES_SHAPE)
_FEATURES_INTEGERS = torch.arange(-2, 6).to(dtype=torch.float32).reshape(1, 8, 1, 1) * _FEATURES_ONES
_RELU_MODULE       = nn.ReLU()

_IN_CHANNELS  = 8
_OUT_CHANNELS = 16
_KERNEL_SIZE  = 3
_PADDING      = 1
_HAS_BIAS     = False
_GRAD_SHAPE   = (1, 16, 200, 200)
_GRAD_ONES    = torch.ones(_GRAD_SHAPE)
_CONV2D_MODULE = nn.Conv2d(in_channels=_IN_CHANNELS, out_channels=_OUT_CHANNELS, kernel_size=_KERNEL_SIZE, padding=_PADDING, bias=_HAS_BIAS)

_LEARNING_RATE_LOW  = 1e-6
_LEARNING_RATE_HIGH = 1e-3

_LOOP_LENGTH = 100


class PACTModulesTest(unittest.TestCase):

    @staticmethod
    def _check_integerisation(reversedfq: torch.Tensor, tq: torch.Tensor) -> bool:
        absdiff = torch.abs(reversedfq - tq)
        cond = bool(torch.all((0.0 <= absdiff) & (absdiff <= 1.0)))  # I know that the absolute value has non-negative codomain, but I state the condition explicitly for readability
        return cond

    def test__pactactivation(self):

        # clip_lo-only
        # create object
        qrangespec = 'binary'
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = ('const', {'a': -0.25, 'b': 0.25})
        pactrelu = PACTReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(pactrelu.clip_lo.requires_grad)
        self.assertFalse(pactrelu.clip_hi.requires_grad)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(pactrelu(x) == _RELU_MODULE(x)))
        self.assertTrue(pactrelu.inplace == _RELU_MODULE.inplace)
        # finalise quantiser parametrisation
        pactrelu.init_qhparams()
        # fake-quantise input
        fqy = pactrelu(_FEATURES_INTEGERS)
        tqy = fqy / (pactrelu.step * pactrelu.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqy, torch.floor(tqy)))
        # backpropagate
        fqy.backward(_FEATURES_ONES)
        clip_lo_grad = torch.sum(_FEATURES_ONES[_FEATURES_INTEGERS < pactrelu.clip_lo])
        clip_hi_grad = torch.sum(_FEATURES_ONES[pactrelu.clip_hi <= _FEATURES_INTEGERS])
        self.assertTrue(pactrelu.clip_lo.grad == (clip_lo_grad - clip_hi_grad))
        self.assertTrue(pactrelu.clip_hi.grad is None)
        pactrelu.clip_lo.data -= _LEARNING_RATE_LOW * pactrelu.clip_lo.grad
        pactrelu.clip_lo.grad = None
        # ...another round to verify that it's working ok...
        fqy = pactrelu(_FEATURES_INTEGERS)
        fqy.backward(_FEATURES_ONES)
        clip_lo_grad = torch.sum(_FEATURES_ONES[_FEATURES_INTEGERS < pactrelu.clip_lo])
        clip_hi_grad = torch.sum(_FEATURES_ONES[pactrelu.clip_hi <= _FEATURES_INTEGERS])
        self.assertTrue(pactrelu.clip_lo.grad == (clip_lo_grad - clip_hi_grad))
        self.assertTrue(pactrelu.clip_hi.grad is None)
        pactrelu.clip_lo.data -= _LEARNING_RATE_LOW * pactrelu.clip_lo.grad
        pactrelu.clip_lo.grad = None
        # freeze clipping bounds
        pactrelu.freeze()
        self.assertFalse(pactrelu.clip_lo.requires_grad)
        self.assertFalse(pactrelu.clip_hi.requires_grad)
        x = torch.randn(_FEATURES_SHAPE)
        x.requires_grad = True
        fqy = pactrelu(x)
        tqy = fqy / (pactrelu.step * pactrelu.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqy, torch.floor(tqy)))
        self.assertTrue(pactrelu.clip_lo.grad is None)
        self.assertTrue(pactrelu.clip_hi.grad is None)
        # thaw clipping bounds
        pactrelu.thaw()
        self.assertTrue(pactrelu.clip_lo.requires_grad)
        self.assertFalse(pactrelu.clip_hi.requires_grad)
        fqy = pactrelu(_FEATURES_INTEGERS)
        fqy.backward(_FEATURES_ONES)
        clip_lo_grad = torch.sum(_FEATURES_ONES[_FEATURES_INTEGERS < pactrelu.clip_lo])
        clip_hi_grad = torch.sum(_FEATURES_ONES[pactrelu.clip_hi <= _FEATURES_INTEGERS])
        self.assertTrue(pactrelu.clip_lo.grad == (clip_lo_grad - clip_hi_grad))
        self.assertTrue(pactrelu.clip_hi.grad is None)
        # induce a crash by bringing clip_lo to the non-negative real range
        pactrelu.clip_lo.data -= _LEARNING_RATE_HIGH * pactrelu.clip_lo.grad
        pactrelu.clip_lo.grad = None
        self.assertRaises(RuntimeError, lambda: pactrelu(_FEATURES_ONES))

        # clip_hi-only
        # create object
        qrangespec = {'bitwidth': 4, 'signed': False}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = ('const', {'a': 0.0, 'b': 2.0})
        pactrelu = PACTReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertFalse(pactrelu.clip_lo.requires_grad)
        self.assertTrue(pactrelu.clip_hi.requires_grad)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(pactrelu(x) == _RELU_MODULE(x)))
        self.assertTrue(pactrelu.inplace == _RELU_MODULE.inplace)
        # finalise quantiser parametrisation
        pactrelu.init_qhparams()
        # fake-quantise input
        fqy = pactrelu(_FEATURES_INTEGERS)
        tqy = fqy / (pactrelu.step * pactrelu.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqy, torch.floor(tqy)))
        # backpropagate
        fqy.backward(_FEATURES_ONES)
        self.assertTrue(pactrelu.clip_lo.grad is None)
        clip_hi_grad = torch.sum(_FEATURES_ONES[pactrelu.clip_hi <= _FEATURES_INTEGERS])
        self.assertTrue(pactrelu.clip_hi.grad == clip_hi_grad)
        pactrelu.clip_hi.data -= _LEARNING_RATE_LOW * pactrelu.clip_hi.grad
        pactrelu.clip_hi.grad = None
        # ...another round to verify that it's working ok...
        fqy = pactrelu(_FEATURES_INTEGERS)
        fqy.backward(_FEATURES_ONES)
        self.assertTrue(pactrelu.clip_lo.grad is None)
        clip_hi_grad = torch.sum(_FEATURES_ONES[pactrelu.clip_hi <= _FEATURES_INTEGERS])
        self.assertTrue(pactrelu.clip_hi.grad == clip_hi_grad)
        pactrelu.clip_hi.data -= _LEARNING_RATE_LOW * pactrelu.clip_hi.grad
        pactrelu.clip_hi.grad = None
        # freeze clipping bounds
        pactrelu.freeze()
        self.assertFalse(pactrelu.clip_lo.requires_grad)
        self.assertFalse(pactrelu.clip_hi.requires_grad)
        x = torch.randn(_FEATURES_SHAPE)
        x.requires_grad = True
        fqy = pactrelu(x)
        tqy = fqy / (pactrelu.step * pactrelu.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqy, torch.floor(tqy)))
        self.assertTrue(pactrelu.clip_lo.grad is None)
        self.assertTrue(pactrelu.clip_hi.grad is None)
        # thaw clipping bounds
        pactrelu.thaw()
        self.assertFalse(pactrelu.clip_lo.requires_grad)
        self.assertTrue(pactrelu.clip_hi.requires_grad)
        fqy = pactrelu(_FEATURES_INTEGERS)
        fqy.backward(_FEATURES_ONES)
        self.assertTrue(pactrelu.clip_lo.grad is None)
        clip_hi_grad = torch.sum(_FEATURES_ONES[pactrelu.clip_hi <= _FEATURES_INTEGERS])
        self.assertTrue(pactrelu.clip_hi.grad == clip_hi_grad)
        pactrelu.clip_hi.data -= _LEARNING_RATE_LOW * pactrelu.clip_hi.grad
        # induce a crash by bringing clip_hi to the non-positive real range
        pactrelu.clip_hi.data -= _LEARNING_RATE_HIGH * pactrelu.clip_hi.grad
        pactrelu.clip_hi.grad = None
        self.assertRaises(RuntimeError, lambda: pactrelu(_FEATURES_ONES))

        # clip_lo and clip_hi
        qrangespec = {'n_levels': 255}
        qgranularityspec = 'per-array'
        qhparamsinitstrategyspec = 'minmax'
        pactrelu = PACTReLU.from_fp_module(_RELU_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertTrue(pactrelu.clip_lo.requires_grad)
        self.assertTrue(pactrelu.clip_hi.requires_grad)
        # does it have ReLU behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(pactrelu(x) == _RELU_MODULE(x)))
        self.assertTrue(pactrelu.inplace == _RELU_MODULE.inplace)
        # warm-up observer and finalise quantiser parametrisation
        self.assertFalse(pactrelu._is_observing)
        pactrelu.start_observing()
        for i in range(_LOOP_LENGTH):
            self.assertTrue(pactrelu._is_observing)
            x = torch.randn(_FEATURES_SHAPE) + 0.01 * i
            _ = pactrelu(x)
        pactrelu.stop_observing()
        self.assertFalse(pactrelu._is_observing)
        # fake-quantise input
        fqy = pactrelu(_FEATURES_INTEGERS)
        tqy = fqy / (pactrelu.step * pactrelu.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqy, torch.floor(tqy)))
        # backpropagate
        fqy.backward(_FEATURES_ONES)
        self.assertFalse(pactrelu.clip_lo.grad is None)
        self.assertFalse(pactrelu.clip_hi.grad is None)
        pactrelu.clip_lo.grad = None
        pactrelu.clip_hi.grad = None
        # freeze clipping bounds
        pactrelu.freeze()
        self.assertFalse(pactrelu.clip_lo.requires_grad)
        self.assertFalse(pactrelu.clip_hi.requires_grad)
        x = torch.randn(_FEATURES_SHAPE)
        x.requires_grad = True
        fqy = pactrelu(x)
        tqy = fqy / (pactrelu.step * pactrelu.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqy, torch.floor(tqy)))
        self.assertTrue(pactrelu.clip_lo.grad is None)
        self.assertTrue(pactrelu.clip_hi.grad is None)
        # thaw clipping bounds
        pactrelu.thaw()
        self.assertTrue(pactrelu.clip_lo.requires_grad)
        self.assertTrue(pactrelu.clip_hi.requires_grad)
        fqy = pactrelu(_FEATURES_INTEGERS)
        fqy.backward(_FEATURES_ONES)
        self.assertFalse(pactrelu.clip_lo.grad is None)
        self.assertFalse(pactrelu.clip_hi.grad is None)
        pactrelu.clip_lo.grad = None
        pactrelu.clip_hi.grad = None

    def test__pactlinear(self):

        # create object
        qrangespec = {'bitwidth': 8, 'signed': True}
        qgranularityspec = 'per-outchannel_weights'
        qhparamsinitstrategyspec = 'meanstd'  # using a statistics-dependent strategy is fundamental, otherwise the quantisers might not be re-computed
        pactc2d = PACTConv2d.from_fp_module(_CONV2D_MODULE, qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        self.assertFalse(pactc2d.clip_lo.requires_grad)
        self.assertFalse(pactc2d.clip_hi.requires_grad)
        # does it have Conv2d behaviour when unquantised?
        x = torch.randn(_FEATURES_SHAPE)
        self.assertTrue(torch.all(pactc2d(x) == _CONV2D_MODULE(x)))
        # finalise quantiser parametrisation
        pactc2d.init_qhparams()
        # fake-quantise weights
        fqw = pactc2d.qweight
        tqw = fqw / (pactc2d.step * pactc2d.scale)
        self.assertTrue(PACTModulesTest._check_integerisation(tqw, torch.floor(tqw)))
        # forward/backward iteration (clipping bounds should change)
        clip_lo_old = pactc2d.clip_lo.data
        clip_hi_old = pactc2d.clip_hi.data
        x = torch.randn(_FEATURES_SHAPE)
        y = pactc2d(x)
        y.backward(_GRAD_ONES)
        pactc2d.weight.data -= _LEARNING_RATE_LOW * pactc2d.weight.grad  # update weights, and therefore also their statistics
        _ = pactc2d.qweight  # trigger quantiser re-computation
        clip_lo_new = pactc2d.clip_lo.data
        clip_hi_new = pactc2d.clip_hi.data
        self.assertFalse(torch.all(clip_lo_new == clip_lo_old))
        self.assertFalse(torch.all(clip_hi_new == clip_hi_old))
        # freeze clipping bounds
        pactc2d.freeze()
        clip_lo_old = pactc2d.clip_lo.data
        clip_hi_old = pactc2d.clip_hi.data
        x = torch.randn(_FEATURES_SHAPE)
        y = pactc2d(x)
        y.backward(_GRAD_ONES)
        pactc2d.weight.data -= _LEARNING_RATE_LOW * pactc2d.weight.grad  # update weights, and therefore also their statistics
        _ = pactc2d.qweight  # trigger quantiser re-computation
        clip_lo_new = pactc2d.clip_lo.data
        clip_hi_new = pactc2d.clip_hi.data
        self.assertTrue(torch.all(clip_lo_new == clip_lo_old))
        self.assertTrue(torch.all(clip_hi_new == clip_hi_old))
        # thaw clipping bounds
        pactc2d.thaw()
        clip_lo_old = pactc2d.clip_lo.data
        clip_hi_old = pactc2d.clip_hi.data
        x = torch.randn(_FEATURES_SHAPE)
        y = pactc2d(x)
        y.backward(_GRAD_ONES)
        pactc2d.weight.data -= _LEARNING_RATE_LOW * pactc2d.weight.grad  # update weights, and therefore also their statistics
        _ = pactc2d.qweight  # trigger quantiser re-computation
        clip_lo_new = pactc2d.clip_lo.data
        clip_hi_new = pactc2d.clip_hi.data
        self.assertFalse(torch.all(clip_lo_new == clip_lo_old))
        self.assertFalse(torch.all(clip_hi_new == clip_hi_old))
