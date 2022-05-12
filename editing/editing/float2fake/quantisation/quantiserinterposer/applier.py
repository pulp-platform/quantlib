import copy
import torch.nn as nn
import torch.fx as fx

from ..qdescription import QDescription, QDescriptionSpecType, resolve_qdescriptionspec
from quantlib.editing.editing.editors.nnmodules import NodesMap
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from quantlib.editing.editing.editors.nnmodules import NNModuleApplier


class QuantiserInterposerApplier(NNModuleApplier):

    def __init__(self,
                 qdescriptionspec: QDescriptionSpecType,
                 pattern:          NNSequentialPattern):

        qdescription = resolve_qdescriptionspec(qdescriptionspec)

        super(QuantiserInterposerApplier, self).__init__(pattern)
        self._qdescription = qdescription

    @property
    def qdescription(self) -> QDescription:
        return self._qdescription

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        """Insert a quantiser between two linear opeartions."""

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_pre  = name_to_match_node['linear_pre']
        node_post = name_to_match_node['linear_post']

        # create the new quantiser
        new_target = id_
        qgranularityspec, qrangespec, qhparamsinitstrategyspec, (mapping, kwargs) = copy.deepcopy(self.qdescription)
        new_module = mapping[nn.Identity](qrangespec=qrangespec,
                                          qgranularityspec=qgranularityspec,
                                          qhparamsinitstrategyspec=qhparamsinitstrategyspec,
                                          **kwargs)

        # add the quantiser to the graph (interposing it between the two linear nodes)
        # We want that after the rewriting each user of `node_pre` reads the
        # output of `new_node` instead; however, in the intermediate state,
        # `new_node` will itself be a user of `node_pre`. Therefore, we need
        # to determine who these users are before `new_node` becomes one of
        # them.
        downstream_nodes = list(node_pre.users)
        assert node_post in downstream_nodes
        # rewrite the graph
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_pre):
            new_node = g.graph.call_module(new_target, args=(node_pre,))
        for u in downstream_nodes:
            u.replace_input_with(node_pre, new_node)

        return g
