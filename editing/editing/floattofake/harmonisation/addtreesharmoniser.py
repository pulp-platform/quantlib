import torch.fx as fx
from typing import List

from quantlib.editing.editing.editors.editors import ApplicationPoint, Rewriter
from quantlib.editing.editing.matching.optrees import OpTree, AddTreeMatcher
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from quantlib.editing.graphs.nn.harmonisedadd import HarmonisedAdd
from quantlib.utils import quantlib_err_header


class AddTreesHarmoniser(Rewriter):

    def __init__(self,
                 algorithm: str,
                 qrangespec: QRangeSpecType,
                 qgranularityspec: QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 force_output_scale: bool = False):

        name = 'AddTreesHarmoniser'
        super(AddTreesHarmoniser, self).__init__(name)

        self._matcher = AddTreeMatcher()

        self._algorithm                = algorithm
        self._qrangespec               = qrangespec
        self._qgranularityspec         = qgranularityspec
        self._qhparamsinitstrategyspec = qhparamsinitstrategyspec
        self._force_output_scale       = force_output_scale

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        candidate_matches = self._matcher.find_application_points(g)
        aps = [ApplicationPoint(rewriter=self, graph=g, apcore=match) for match in candidate_matches]
        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:

        # Were the application points computed by this `Rewriter`, and on the target `fx.GraphModule`?
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        # create harmoniser for the harmonisation context represented by the `OpTree`
        assert isinstance(ap.apcore, OpTree)
        harmoniser = HarmonisedAdd(ap=ap.apcore,
                                   algorithm=self._algorithm,
                                   qrangespec=self._qrangespec,
                                   qgranularityspec=self._qgranularityspec,
                                   qhparamsinitstrategyspec=self._qhparamsinitstrategyspec,
                                   force_output_scale=self._force_output_scale)

        # insert harmoniser in its harmonisation context
        # compute identity information for the new code
        self._counter += 1
        new_target = '_'.join([self._name.upper(), harmoniser.__class__.__name__.upper(), str(self._counter)])
        # add module to the graph
        g.add_submodule(new_target, harmoniser)
        with g.graph.inserting_before(ap.apcore.root):
            new_node = g.graph.call_module(new_target, args=ap.apcore.inbound_frontier)
        ap.apcore.root.replace_all_uses_with(new_node)  # attach the module to the previous users of the tree's end node
        # remove dead code
        for node in ap.apcore.nodes:
            g.graph.erase_node(node)

        return g
