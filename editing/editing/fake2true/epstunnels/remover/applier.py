import itertools
import torch.fx as fx

from .applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Applier


class EpsTunnelRemoverApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: EpsTunnelNode, id_: str) -> fx.GraphModule:

        node = ap.node

        # the `fx.Node` is functionally equivalent to the identity, so we connect its (unique) input to all the outputs
        predecessors = {p for p in node.all_input_nodes}  # upstream
        assert len(predecessors) == 1
        successors = {s for s in node.users}  # downstream
        for p, s in itertools.product(predecessors, successors):
            s.replace_input_with(node, p)

        g.delete_submodule(node.target)
        g.graph.erase_node(node)

        return g
