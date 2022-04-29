import networkx as nx
import torch.fx as fx
from typing import List, Dict

from .fx2nx import NXFXGraph
from .pattern import GenericGraphPattern


NodesMap = Dict[fx.Node, fx.Node]


class GenericGraphMatcher(object):

    def __init__(self, pattern: GenericGraphPattern):
        self._pattern = pattern

    @property
    def pattern(self) -> GenericGraphPattern:
        return self._pattern

    def find(self, data_gm: fx.GraphModule) -> List[NodesMap]:

        data_nxg = NXFXGraph.from_fx_graph(data_gm.graph)

        matcher = nx.algorithms.isomorphism.DiGraphMatcher(data_nxg, self.pattern.pattern_nxg, node_match=self.pattern.get_node_matching_function(data_gm))
        nx2nx_matches = list(matcher.subgraph_isomorphisms_iter())

        fx2fx_matches = [{self.pattern.pattern_nxg.nodes[pn]['fx']: data_nxg.nodes[dn]['fx'] for dn, pn in match.items()} for match in nx2nx_matches]

        return fx2fx_matches
