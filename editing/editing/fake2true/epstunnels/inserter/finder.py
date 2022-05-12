import torch.fx as fx
from typing import List

from .applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Finder
from quantlib.editing.editing.fake2true.annotation import EpsPropagator


class EpsTunnelInserterFinder(Finder):

    def find(self, g: fx.GraphModule) -> List[EpsTunnelNode]:
        aps = [EpsTunnelNode(n) for n in g.graph.nodes if EpsPropagator.returns_qtensor(n)]
        return aps

    def check_aps_commutativity(self, aps: List[EpsTunnelNode]) -> bool:
        return len(aps) == len(set(map(lambda ap: ap.node, aps)))
