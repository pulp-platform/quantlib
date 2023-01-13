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


_TARGET_SHAPE                = (4, 16, 8, 8)
_BROADCASTING_SHAPE_TRIVIAL  = (1,)
_BROADCASTING_SHAPE_GRANULAR = (4, 1, 1, 1)
_INCONSISTENT_NDIM           = (4, 16, 8)
_INCONSISTENT_SHAPE          = (8, 16, 4, 4)
_CONSISTENT_SHAPE            = (4, 16, 4, 4)

_LOOP_LENGTH = 100


class ObserverTest(unittest.TestCase):

    def test_loop_per_array(self):

        # zero-dimensional arrays are not supported
        qgranularity = resolve_qgranularityspec('per-array')
        observer = MinMaxMeanVarObserver(subpopulation_dims=qgranularity)
        t = torch.randn(1).squeeze()
        self.assertRaises(ValueError, lambda: observer.update(t))

        # inconsistent number of dimensions across updates
        qgranularity = resolve_qgranularityspec('per-array')
        observer = MinMaxMeanVarObserver(subpopulation_dims=qgranularity)
        t = torch.randn(_TARGET_SHAPE)
        observer.update(t)
        t = torch.randn(_INCONSISTENT_NDIM)  # `t.ndim == 3` as opposed to the expected `t.ndim == 4`
        self.assertRaises(ValueError, lambda: observer.update(t))

        # consistent shapes across updates (verify broadcastability)
        qgranularity = resolve_qgranularityspec('per-array')
        observer = MinMaxMeanVarObserver(subpopulation_dims=qgranularity)
        for _ in range(0, _LOOP_LENGTH):
            t = torch.randn(_TARGET_SHAPE)
            observer.update(t)
        self.assertTrue(observer.broadcasting_shape == _BROADCASTING_SHAPE_TRIVIAL)
        self.assertTrue(observer.n.shape    == observer.broadcasting_shape)
        self.assertTrue(observer.min.shape  == observer.broadcasting_shape)
        self.assertTrue(observer.max.shape  == observer.broadcasting_shape)
        self.assertTrue(observer.mean.shape == observer.broadcasting_shape)
        self.assertTrue(observer.var.shape  == observer.broadcasting_shape)

    def test_loop_per_channel_features(self):

        # inconsistent shapes across updates
        qgranularity = resolve_qgranularityspec('per-outchannel_weights')
        observer = MinMaxMeanVarObserver(subpopulation_dims=qgranularity)
        t = torch.randn(_TARGET_SHAPE)
        observer.update(t)
        t = torch.randn(_INCONSISTENT_SHAPE)
        self.assertRaises(ValueError, lambda: observer.update(t))

        # consistent shapes across updates (broadcastability is preserved)
        qgranularity = resolve_qgranularityspec('per-outchannel_weights')
        observer = MinMaxMeanVarObserver(subpopulation_dims=qgranularity)
        t = torch.randn(_TARGET_SHAPE)
        observer.update(t)
        t = torch.randn(_CONSISTENT_SHAPE)
        self.assertTrue(observer.broadcasting_shape == _BROADCASTING_SHAPE_GRANULAR)
        self.assertTrue(observer.n.shape    == observer.broadcasting_shape)
        self.assertTrue(observer.min.shape  == observer.broadcasting_shape)
        self.assertTrue(observer.max.shape  == observer.broadcasting_shape)
        self.assertTrue(observer.mean.shape == observer.broadcasting_shape)
        self.assertTrue(observer.var.shape  == observer.broadcasting_shape)
