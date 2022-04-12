# 
# performance_unit_tests.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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

import time
import numpy as np

from typing import Tuple

import torch.nn as nn
from .create_tensors import TensorGenerator


def profile(x_gen:    TensorGenerator,
            module:   nn.Module,
            grad_gen: TensorGenerator,
            T:        int = 1000) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Profile the time performance of a PyTorch module.

    Given a PyTorch module, execute its forward and backward passes as many
    times as specified by `T`. Then, return summary sample statistics (mean
    and standard deviation) about execution times.
    """

    measurements_forward  = list()
    measurements_backward = list()

    for t in range(0, T):

        x = next(x_gen)
        x.requires_grad = True
        sf  = time.time()
        y   = module(x)
        ef  = time.time()
        dtf = ef - sf

        yg = next(grad_gen)
        sb  = time.time()
        y.backward(yg)
        eb  = time.time()
        dtb = eb - sb

        measurements_forward.append(dtf)
        measurements_backward.append(dtb)

    forward_mean   = np.mean(measurements_forward)
    forward_stddev = np.std(measurements_forward)

    backward_mean   = np.mean(measurements_backward)
    backward_stddev = np.std(measurements_backward)

    return (forward_mean, forward_stddev), (backward_mean, backward_stddev)

