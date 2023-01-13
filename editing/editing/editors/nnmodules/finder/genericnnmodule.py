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

import networkx as nx
import torch.fx as fx
from typing import List

from ..applicationpoint import NodesMap
from ..pattern.genericnnmodule.nxfxgraph import NXFXGraph
from ..pattern import GenericNNModulePattern
from .base import NNModuleMatcher


class GenericGraphMatcher(NNModuleMatcher):

    def __init__(self, pattern: GenericNNModulePattern):
        if not isinstance(pattern, GenericNNModulePattern):
            raise TypeError
        super(GenericGraphMatcher, self).__init__(pattern)

    def find(self, data_gm: fx.GraphModule) -> List[NodesMap]:

        # push-forward the data `fx.Graph` to a NetworkX graph
        data_nxg = NXFXGraph.from_fx_graph(data_gm.graph)

        # use NetworkX's sub-graph isomorphism routine to identify candidate matches
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(data_nxg, self.pattern.nxg, node_match=self.pattern.get_nxnode_matching_function(data_gm))
        nx2nx_matches = list(matcher.subgraph_isomorphisms_iter())

        # pull-back matching between NetworkX nodes to a matching over `fx.Node`s
        fx2fx_matches = [NodesMap([(self.pattern.nxg.nodes[pn]['fx'], data_nxg.nodes[dn]['fx']) for dn, pn in match.items()]) for match in nx2nx_matches]

        return fx2fx_matches

    def check_aps_commutativity(self, aps: List[NodesMap]) -> bool:
        return True  # TODO: implement the check!
