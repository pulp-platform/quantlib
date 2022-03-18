from collections import OrderedDict
import torch.nn as nn
import torch.fx as fx
from typing import List

from quantlib.newediting.editing.editors.editors import ApplicationPoint, Rewriter
from quantlib.newediting.editing.matching.lineargraphs import LinearGraphMatcher
from quantlib.newediting.graphs import qmodule_symbolic_trace
from quantlib.newediting.graphs import FXOPCODE_CALL_MODULE, nnmodule_from_fxnode
from quantlib.newutils import quantlib_err_header


linearbn_pattern = OrderedDict([
    ('linear', nn.Linear(in_features=1, out_features=1, bias=True)),
    ('bn',     nn.BatchNorm1d(num_features=1)),
])


class LinearBNBiasFolder(Rewriter):

    def __init__(self):

        name = 'LinearBNBiasFolder'
        super(LinearBNBiasFolder, self).__init__(name)

        self._matcher = LinearGraphMatcher(symbolic_trace_fn=qmodule_symbolic_trace, pattern_module=nn.Sequential(linearbn_pattern))
        self._patternname_2_patternnode = {n.target: n for n in filter(lambda n: (n.op in FXOPCODE_CALL_MODULE) and (n.target in linearbn_pattern.keys()), self._matcher.pattern_gm.graph.nodes)}

        self._linear_node = self._patternname_2_patternnode['linear']
        self._bn_node     = self._patternname_2_patternnode['bn']

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:

        candidate_matches = self._matcher.find(g)
        candidate_matches = list(filter(lambda match: nnmodule_from_fxnode(match.nodes_map[self._linear_node], g).bias is not None, candidate_matches))

        aps = [ApplicationPoint(rewriter=self, graph=g, apcore=match.nodes_map) for match in candidate_matches]
        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:

        # Were the application points computed by this `Rewriter`, and on the target `fx.GraphModule`?
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        bias = nnmodule_from_fxnode(ap.apcore[self._linear_node], g).bias.data.clone().detach()
        nnmodule_from_fxnode(ap.apcore[self._linear_node], g).bias = None
        nnmodule_from_fxnode(ap.apcore[self._bn_node], g).running_mean.data -= bias

        return g
