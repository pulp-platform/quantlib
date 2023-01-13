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

"""The data structure used by ``ModuleWiseFinder``s to identify application
points.
"""

from collections import OrderedDict
import torch.nn as nn
from typing import Tuple


class NameToModule(OrderedDict):
    """A map from symbolic names to ``nn.Module`` objects."""

    @staticmethod
    def split_path_to_target(target: str) -> Tuple[str, str]:
        """Separate an ``nn.Module``'s name from its parent's qualified name.

        The hierarchy of ``nn.Module``s that composes a PyTorch network is
        captured by the module names through dotted notation (i.e., the names
        are *qualified*).
        """
        *ancestors, child = target.rsplit('.')
        path_to_parent = '.'.join(ancestors) if len(ancestors) > 0 else ''
        return path_to_parent, child

    def __setitem__(self, name: str, module: nn.Module):
        """We enforce type checking by overwriting the parent class's method."""

        if not isinstance(name, str):
            raise TypeError
        if not isinstance(module, nn.Module):
            raise TypeError

        super(NameToModule, self).__setitem__(name, module)
