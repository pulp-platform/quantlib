from functools import partial
import torch.fx as fx
from typing import Callable, Dict

from .nxfxgraph import NXFXGraph
from ..base import NNModulePattern, SymbolicTraceFnType, NNModuleWithCheckers


class GenericNNModulePattern(NNModulePattern):

    def __init__(self,
                 module_with_checkers:    NNModuleWithCheckers,
                 symbolic_trace_fn:       SymbolicTraceFnType):

        super(GenericNNModulePattern, self).__init__(module_with_checkers, symbolic_trace_fn)
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
