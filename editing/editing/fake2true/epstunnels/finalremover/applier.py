import itertools
import torch.fx as fx
import torch

from ..remover.applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Applier


class FinalEpsTunnelRemoverApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: EpsTunnelNode, id_: str) -> fx.GraphModule:

        node = ap.node

        # the `fx.Node` is functionally equivalent to the identity, so we connect its (unique) input to all the outputs
        predecessors = {p for p in node.all_input_nodes}  # upstream
        assert len(predecessors) == 1
        successors = {s for s in node.users}  # downstream
        assert len(successors) == 1
        for p, s in itertools.product(predecessors, successors):
            s.replace_input_with(node, p)
        torch.set_printoptions(precision=16)
        print("[FinalEpsTunnelRemover] %s: removing EpsTunnel with scaling factor %s" % (s, g.get_submodule(node.target).eps_out/g.get_submodule(node.target).eps_in))
        print("[FinalEpsTunnelRemover] %s: outputs will need to be scaled *externally* to maintain program semantics.")
        torch.set_printoptions()

        g.delete_submodule(node.target)
        g.graph.erase_node(node)

        return g
