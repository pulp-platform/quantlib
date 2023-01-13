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

from ..pattern.base import NNModulePattern
from ..finder.base import NNModuleMatcher
from ..applier import NNModuleApplier
from quantlib.editing.editing.editors.base import Rewriter


class NNModuleRewriter(Rewriter):

    def __init__(self,
                 name:    str,
                 pattern: NNModulePattern,
                 finder:  NNModuleMatcher,
                 applier: NNModuleApplier):

        if not ((pattern is finder.pattern) and (pattern is applier.pattern)):
            raise ValueError

        super(NNModuleRewriter, self).__init__(name, pattern.symbolic_trace_fn, finder, applier)
        self._pattern = pattern

    @property
    def pattern(self) -> NNModulePattern:
        return self._pattern
