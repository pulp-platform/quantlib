import torch
import torch.nn as nn
import torch.fx as fx
from typing import Union


from quantlib.editing.fx.annotations.eps import EpsAnnotator


class EpsTunnel(nn.Module):

    def __init__(self, eps: Union[float, torch.Tensor]):
        super().__init__()

        # canonicalise input
        if isinstance(eps, float):
            eps = torch.Tensor([eps])
        else:
            if eps.numel() != 1:
                raise ValueError("[QuantLab] `EpsTunnel` should be initialised with a unique quantum value; instead, I received {}".format(eps))

        self.eps_in = eps
        self.eps_out = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eps_in != self.eps_out:
            x = self.eps_out * (x / self.eps_in)
        return x


class EpsTunnelAdder(object):

    def __init__(self, gm: fx.GraphModule):
        self._gm = gm

    @property
    def gm(self) -> fx.GraphModule:
        return self._gm

    def apply(self):

        for n in self.gm.graph.nodes:

            if EpsAnnotator.returns_qtensor(n):

                downstream_ops = {u for u in n.users if EpsAnnotator.is_eps_annotated(u)}

                for u in downstream_ops:
                    new_target = "({})->({})".format(n.target, u.target)
                    new_module = EpsTunnel(n.meta['eps'])
                    self.gm.add_submodule(new_target, new_module)
                    with self.gm.graph.inserting_before(u):
                        new_node = self.gm.graph.call_module(new_target, args=(n,))
                    u.replace_input_with(n, new_node)

        self.gm.recompile()
