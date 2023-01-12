import torch.fx as fx

from .applicationpoint import EpsTunnelNode
# from quantlib.editing.editing.fake2true.annotation.epspropagator.propagationrules import is_eps_annotated  # TODO: see below
from quantlib.editing.editing.editors import Applier
from quantlib.editing.graphs.nn import EpsTunnel


class EpsTunnelInserterApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: EpsTunnelNode, id_: str) -> fx.GraphModule:

        node = ap.node

        # create a new `EpsTunnel` ...
        new_target = id_
        new_module = EpsTunnel(node.meta['eps'])

        # ... and place it immediately after the `fx.Node` emitting a fake-quantised `torch.Tensor`
        downstream_nodes = list(node.users)
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node):
            new_node = g.graph.call_module(new_target, args=(node,))
        for u in downstream_nodes:
            u.replace_input_with(node, new_node)

        # TODO: the graph rewriting logic is the same as that used by
        #       `QuantiserInterposerApplier`. Is there a smart way to define a
        #       shared abstraction?

        # if the `fx.Node` is used by multiple downstream nodes, push a different `EpsTunnel` copy down each path
        # downstream_nodes = {u for u in new_node.users if is_eps_annotated(u)}  # TODO: why did I do this?...
        downstream_nodes = list(new_node.users)
        if len(downstream_nodes) > 1:

            local_counter: int = 0
            for u in downstream_nodes:

                new_target_copy = id_ + f'_{str(local_counter)}_'
                new_module_copy = EpsTunnel(node.meta['eps'])

                g.add_submodule(new_target_copy, new_module_copy)
                with g.graph.inserting_before(u):
                    new_node_copy = g.graph.call_module(new_target_copy, args=(new_node,))
                u.replace_input_with(new_node, new_node_copy)

                local_counter += 1

        return g
