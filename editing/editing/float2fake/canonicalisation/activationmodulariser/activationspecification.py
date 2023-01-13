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

"""The object guiding ``ActivationFinder``s when filtering ``fx.Node``s, and
``ActivationReplacer``s when building modular activations.
"""

import torch
import torch.nn as nn
from typing import NamedTuple, Tuple, Callable, Type


ActivationTarget = Callable[[torch.Tensor], torch.Tensor]


class NonModularTargets(NamedTuple):
    """Collections of ``fx.Node`` targets representing invocations of
    activation functions based on PyTorch's non-modular API.
    """
    inplace:    Tuple[ActivationTarget, ...]  # should the replacement be called inplace?
    noninplace: Tuple[ActivationTarget, ...]  # should the replacement not be called inplace?

    def __contains__(self, item):
        return (item in self.inplace) or (item in self.noninplace)


class ActivationSpecification(NamedTuple):
    """Container attaching the PyTorch modular API for an activation function
    to its non-modular counterparts.
    """
    module_class: Type[nn.Module]
    targets:      NonModularTargets
