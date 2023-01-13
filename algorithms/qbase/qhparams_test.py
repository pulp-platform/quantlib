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

from .qhparams import create_qhparams
from .qhparams import get_scale
from .qhparams import get_zero_scale
from .qhparams import get_clipping_bounds
from .qrange import IMPLICIT_STEP
from .qrange import resolve_qrangespec


class QHParamsTest(unittest.TestCase):

    def test_create_qhparams(self):

        def condition(reference: torch.Tensor, to_be_tested: torch.Tensor) -> bool:
            cond_ndim  = to_be_tested.ndim == reference.ndim
            cond_numel = to_be_tested.numel() == reference.numel()
            cond_value = (to_be_tested == reference) or (torch.isnan(to_be_tested) and torch.isnan(reference))
            passed = cond_ndim and cond_numel and cond_value
            return passed

        # unspecified offset
        dict_ = {'n_levels': 3}  # I must generate a correct input for `init_qhparams`; the correctness of `QRange` resolution is tested elsewhere.
        qrange = resolve_qrangespec(dict_)
        reference_zero     = torch.Tensor([float('nan')])
        reference_n_levels = torch.Tensor([dict_['n_levels']])
        reference_step     = torch.Tensor([IMPLICIT_STEP])
        reference_scale    = torch.Tensor([float('nan')])
        zero, n_levels, step, scale = create_qhparams(qrange)
        self.assertTrue(condition(reference_zero, zero))
        self.assertTrue(condition(reference_n_levels, n_levels))
        self.assertTrue(condition(reference_step, step))
        self.assertTrue(condition(reference_scale, scale))

        # specified offset
        dict_ = {'bitwidth': 8, 'signed': True}
        qrange = resolve_qrangespec(dict_)
        reference_zero     = torch.Tensor([-2**(dict_['bitwidth'] - 1)])
        reference_n_levels = torch.Tensor([2**dict_['bitwidth']])
        reference_step     = torch.Tensor([IMPLICIT_STEP])
        reference_scale    = torch.Tensor([float('nan')])
        zero, n_levels, step, scale = create_qhparams(qrange)
        self.assertTrue(condition(reference_zero, zero))
        self.assertTrue(condition(reference_n_levels, n_levels))
        self.assertTrue(condition(reference_step, step))
        self.assertTrue(condition(reference_scale, scale))

    def test_get_scale(self):

        # incorrect bounds (a not strictly lower than b)
        target_shape = (1, 16, 1, 1)
        a = torch.randn(target_shape)
        b = a - torch.rand(target_shape)
        dict_ = {'bitwidth': 8, 'signed': True}
        qrange = resolve_qrangespec(dict_)
        zero, n_levels, step, _ = create_qhparams(qrange)
        zero     = torch.ones(target_shape) * zero
        n_levels = torch.ones(target_shape) * n_levels
        step     = torch.ones(target_shape) * step
        self.assertRaises(ValueError, lambda: get_scale(a, b, zero, n_levels, step))

        # correct bounds
        target_shape = (1, 16, 1, 1)
        a = torch.randn(target_shape)
        b = a + torch.rand(target_shape)
        dict_ = {'bitwidth': 8, 'signed': True}
        qrange = resolve_qrangespec(dict_)
        zero, n_levels, step, _ = create_qhparams(qrange)
        zero     = torch.ones(target_shape) * zero
        n_levels = torch.ones(target_shape) * n_levels
        step     = torch.ones(target_shape) * step
        scale = get_scale(a, b, zero, n_levels, step)
        self.assertTrue(torch.all(scale > 0))  # quanta should always be positive

    def test_get_zero_scale(self):

        # incorrect bounds (a not strictly lower than b)
        target_shape = (1, 16, 1, 1)
        a = torch.randn(target_shape)
        b = a - torch.rand(target_shape)
        dict_ = {'n_levels': 3}
        qrange = resolve_qrangespec(dict_)
        _, n_levels, step, _ = create_qhparams(qrange)
        n_levels = torch.ones(target_shape) * n_levels
        step     = torch.ones(target_shape) * step
        self.assertRaises(ValueError, lambda: get_zero_scale(a, b, n_levels, step))

        # correct bounds
        target_shape = (1, 16, 1, 1)
        a = torch.randn(target_shape)
        b = a + torch.rand(target_shape)
        dict_ = {'n_levels': 3}
        qrange = resolve_qrangespec(dict_)
        _, n_levels, step, _ = create_qhparams(qrange)
        n_levels = torch.ones(target_shape) * n_levels
        step     = torch.ones(target_shape) * step
        zero, scale = get_zero_scale(a, b, n_levels, step)
        self.assertTrue(torch.all(zero == torch.floor(zero)))  # check that all offsets are integers embedded into the floating-point range
        self.assertTrue(torch.all(scale > 0))                  # quanta should always be positive

    def test_get_clipping_bounds(self):

        target_shape = (1, 16, 1, 1)
        a = torch.randn(target_shape)
        b = a + torch.rand(target_shape)
        dict_ = {'bitwidth': 8, 'signed': True}
        qrange = resolve_qrangespec(dict_)
        zero, n_levels, step, _ = create_qhparams(qrange)
        zero     = torch.ones(target_shape) * zero
        n_levels = torch.ones(target_shape) * n_levels
        step     = torch.ones(target_shape) * step
        scale = get_scale(a, b, zero, n_levels, step)
        clip_lo, clip_hi = get_clipping_bounds(zero, n_levels, step, scale)
        self.assertTrue(torch.all(clip_lo < clip_hi))
