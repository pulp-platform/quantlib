import torch
import torch.fx as fx
from typing import List

from .finder import EpsTunnelConstructFinder
from quantlib.newediting.editing.editors import ApplicationPoint, Rewriter


class EpsTunnelConstructSimplifier(Rewriter):

    def __init__(self):
        name = 'EpsTunnelConstructSimplifier'
        super(EpsTunnelConstructSimplifier, self).__init__(name)

        self._finder = EpsTunnelConstructFinder()

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        return list(map(lambda cc: ApplicationPoint(rewriter=self, graph=g, core=cc), self._finder.find(g)))

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        cc = ap.core
        inbound_eps_tunnels  = tuple(map(lambda n: g.get_submodule(target=n.target), cc.backward))
        outbound_eps_tunnels = tuple(map(lambda n: g.get_submodule(target=n.target), cc.forward))

        for m in inbound_eps_tunnels:
            m.set_eps_out(torch.ones_like(m.eps_out))
        for m in outbound_eps_tunnels:
            m.set_eps_in(torch.ones_like(m.eps_in))

        return g
