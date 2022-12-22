import torch
import torch.fx as fx
from typing import List

from ..remover.applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Finder
from quantlib.editing.graphs.fx import FXOpcodeClasses
from quantlib.editing.graphs.nn import EpsTunnel


class FinalEpsTunnelRemoverFinder(Finder):

    def find(self, g: fx.GraphModule) -> List[EpsTunnelNode]:

        # find `EpsTunnel` `fx.Node`s
        module_nodes = filter(lambda n: (n.op in FXOpcodeClasses.CALL_MODULE.value), g.graph.nodes)

        # select only output nodes
        epstunnels = list(filter(lambda n: isinstance(g.get_submodule(target=n.target), EpsTunnel), module_nodes))  # since we consume the `filter` generator twice in the next lines, we must ensure that it does not get empty after the first consumption
        singleusertunnels = list(filter(lambda n: len(n.users) == 1, epstunnels))
        outputtunnels = list(filter(lambda n: list(n.users.keys())[0].op == "output", singleusertunnels))

        return [EpsTunnelNode(n) for n in list(outputtunnels)]

    def check_aps_commutativity(self, aps: List[EpsTunnelNode]) -> bool:
        return len(aps) == len(set(map(lambda ap: ap.node, aps)))
