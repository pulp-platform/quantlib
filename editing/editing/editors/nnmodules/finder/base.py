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

import torch.fx as fx
from typing import List

from ..applicationpoint import NodesMap
from ..pattern.base import NNModulePattern
from quantlib.editing.editing.editors.base import Finder


class NNModuleMatcher(Finder):
    """A class to find application points by pattern-matching."""

    def __init__(self, pattern: NNModulePattern):
        super(NNModuleMatcher, self).__init__()
        self._pattern = pattern

    @property
    def pattern(self) -> NNModulePattern:
        return self._pattern

    def find(self, g: fx.GraphModule) -> List[NodesMap]:
        raise NotImplementedError

    def check_aps_commutativity(self, aps: List[NodesMap]) -> bool:
        raise NotImplementedError
