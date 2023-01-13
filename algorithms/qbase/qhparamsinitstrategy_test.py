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

from .qgranularity import resolve_qgranularityspec
from .observer import MinMaxMeanVarObserver
from .qhparamsinitstrategy import resolve_qhparamsinitstrategyspec


_TARGET_SHAPE = (4, 16, 8, 8)

_ARRAY_GRANULARITY              = resolve_qgranularityspec('per-array')
_OUTCHANNEL_WEIGHTS_GRANULARITY = resolve_qgranularityspec('per-outchannel_weights')

_BROADCASTING_SHAPE_TRIVIAL  = (1,)
_BROADCASTING_SHAPE_GRANULAR = (4, 1, 1, 1)


class QHparamsInitStrategyTest(unittest.TestCase):

    def test_const_init_strategy(self):

        def condition(reference: torch.Tensor, to_be_tested: torch.Tensor) -> bool:
            cond_shape = to_be_tested.shape == reference.shape
            cond_value = torch.all(to_be_tested == reference)
            return cond_shape and cond_value

        # non-supported keys
        spec = ('const', {'mid': 0.0})
        self.assertRaises(ValueError, lambda: resolve_qhparamsinitstrategyspec(spec))

        # wrong key values
        spec = ('const', {'a': 2.0, 'b': -2.0})
        self.assertRaises(ValueError, lambda: resolve_qhparamsinitstrategyspec(spec))

        # default keys ("cold" observer)
        spec = 'const'
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_ARRAY_GRANULARITY)
        reference_a = torch.ones(_BROADCASTING_SHAPE_TRIVIAL) * qhparamsinitstrategy.default_kwargs['a']
        reference_b = torch.ones(_BROADCASTING_SHAPE_TRIVIAL) * qhparamsinitstrategy.default_kwargs['b']
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(condition(reference_a, a))
        self.assertTrue(condition(reference_b, b))
        self.assertTrue(torch.all(a < b))

        # custom keys ("cold" observer)
        kwargs = {'a': -2.0, 'b': 2.0}
        spec = ('const', kwargs)
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_ARRAY_GRANULARITY)
        reference_a = torch.ones(_BROADCASTING_SHAPE_TRIVIAL) * kwargs['a']
        reference_b = torch.ones(_BROADCASTING_SHAPE_TRIVIAL) * kwargs['b']
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(condition(reference_a, a))
        self.assertTrue(condition(reference_b, b))
        self.assertTrue(torch.all(a < b))

        # default keys ("warmed-up" observer)
        spec = 'const'
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        observer.update(torch.randn(_TARGET_SHAPE))
        reference_a = torch.ones(_BROADCASTING_SHAPE_GRANULAR) * qhparamsinitstrategy.default_kwargs['a']
        reference_b = torch.ones(_BROADCASTING_SHAPE_GRANULAR) * qhparamsinitstrategy.default_kwargs['b']
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(condition(reference_a, a))
        self.assertTrue(condition(reference_b, b))
        self.assertTrue(torch.all(a < b))

        # custom keys ("warmed-up" observer)
        kwargs = {'a': -2.0, 'b': 2.0}
        spec = ('const', kwargs)
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        observer.update(torch.randn(_TARGET_SHAPE))
        reference_a = torch.ones(_BROADCASTING_SHAPE_GRANULAR) * kwargs['a']
        reference_b = torch.ones(_BROADCASTING_SHAPE_GRANULAR) * kwargs['b']
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(condition(reference_a, a))
        self.assertTrue(condition(reference_b, b))
        self.assertTrue(torch.all(a < b))

    def test_minmax_init_strategy(self):

        # non-supported keys
        spec = ('minmax', {'mid': 0.0})
        self.assertRaises(ValueError, lambda: resolve_qhparamsinitstrategyspec(spec))

        # "cold" observer
        spec = 'minmax'
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        self.assertRaises(RuntimeError, lambda: qhparamsinitstrategy.get_a_b(observer))

        # "warmed-up" observer
        spec = 'minmax'
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        t = torch.randn(_TARGET_SHAPE)
        observer.update(t)
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(a.shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(b.shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(torch.all(a < b))

    def test_meanstd_init_strategy(self):

        # non-supported keys
        spec = ('meanstd', {'kurthosis': 0.25})
        self.assertRaises(ValueError, lambda: resolve_qhparamsinitstrategyspec(spec))

        # wrong key values
        spec = ('meanstd', {'n_std': 0})
        self.assertRaises(ValueError, lambda: resolve_qhparamsinitstrategyspec(spec))

        # default keys ("cold" observer)
        spec = 'meanstd'
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        self.assertRaises(RuntimeError, lambda: qhparamsinitstrategy.get_a_b(observer))

        # custom keys ("cold" observer)
        spec = ('meanstd', {'n_std': 2})
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        self.assertRaises(RuntimeError, lambda: qhparamsinitstrategy.get_a_b(observer))

        # default keys ("warmed-up" observer)
        spec = 'meanstd'
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        observer.update(torch.randn(_TARGET_SHAPE))
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(a.shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(b.shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(torch.all(a < b))

        # custom keys ("warmed-up" observer)
        spec = ('meanstd', {'n_std': 2})
        qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(spec)
        observer = MinMaxMeanVarObserver(_OUTCHANNEL_WEIGHTS_GRANULARITY)
        observer.update(torch.randn(_TARGET_SHAPE))
        a, b = qhparamsinitstrategy.get_a_b(observer)
        self.assertTrue(a.shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(b.shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(torch.all(a < b))

    def test_unsupported_init_strategy(self):
        spec = 'MINMAXSQRT'
        self.assertRaises(ValueError, lambda: resolve_qhparamsinitstrategyspec(spec))
