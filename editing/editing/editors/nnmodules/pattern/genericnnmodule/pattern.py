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

from functools import partial
import torch.fx as fx
from typing import Callable, Dict

from ..base import NNModuleWithCheckers
from ..base import NNModulePattern
from .nxfxgraph import NXFXGraph
from quantlib.editing.graphs.fx import SymbolicTraceFnType


class GenericNNModulePattern(NNModulePattern):

    def __init__(self,
                 symbolic_trace_fn:    SymbolicTraceFnType,
                 module_with_checkers: NNModuleWithCheckers):

        super(GenericNNModulePattern, self).__init__(symbolic_trace_fn, module_with_checkers)
        self._nxg: NXFXGraph = NXFXGraph.from_fx_graph(self.fxg)

    @property
    def nxg(self) -> NXFXGraph:
        return self._nxg

    def get_nxnode_matching_function(self, data_gm: fx.GraphModule) -> Callable[[Dict, Dict], bool]:  # NetworkX nodes are implemented as dictionaries
        """Convert the pattern's semantic check between ``fx.Node``s into a
        semantic check between NetworkX nodes.

        This function returns a function to compare pairs of NetworkX nodes.
        This comparer can be passed to NetworkX's sub-graph isomorphism
        routine to inform and accelerate pattern matching; see the
        implementation of the ``find`` method in ``GenericGraphMatcher``.

        """

        fn = partial(NNModulePattern.check_node_attributes, **{'pattern': self, 'data_gm': data_gm})

        def node_match_nx(dn: Dict, pn: Dict) -> bool:  # NetworkX nodes are implemented as dictionaries
            return fn(pn=pn['fx'], dn=dn['fx'])

        return node_match_nx
