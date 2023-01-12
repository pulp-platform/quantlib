from torch import fx
import torch.nn as nn
import quantlib.editing.graphs as qg
import quantlib.editing.editing as qe

class BadDropoutTemplate(nn.Module):
    
    # opcode       name     target    args        kwargs
    # -----------  -------  --------  ----------  --------
    # placeholder  x        x         ()          {}
    # call_module  dropout  dropout   (x,)        {}
    # output       output   output    (dropout,)  {}

    def __init__(self):
        super(BadDropoutTemplate, self).__init__()
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        return self.dropout(x)

class DropoutRemoverApplier(qe.editors.nnmodules.NNModuleApplier):

    def __init__(self, pattern: qe.editors.nnmodules.GenericNNModulePattern):
        super(DropoutRemoverApplier, self).__init__(pattern)

    def _apply(self, g: fx.GraphModule, ap: qe.editors.nnmodules.NodesMap, id_: str) -> fx.GraphModule:

        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_dropout = name_to_match_node['dropout']

        # create the new module
        new_target = id_
        new_module = nn.Identity()

        # add the new module to graph
        g.add_submodule(new_target, new_module)
        node_input  = next(iter(node_dropout.all_input_nodes))
        node_output = node_dropout.next
        node_output.replace_input_with(node_dropout, node_input)

        # ...and delete the old operation
        g.delete_submodule(node_dropout.target)
        g.graph.erase_node(node_dropout)

        return g

class DropoutRemover(qe.editors.nnmodules.NNModuleRewriter):

    def __init__(self):
        # create pattern
        dropout_withcheckers = qe.editors.nnmodules.NNModuleWithCheckers(BadDropoutTemplate(), {})
        dropout_pattern = qe.editors.nnmodules.GenericNNModulePattern(qg.fx.quantlib_symbolic_trace, dropout_withcheckers)
        # create matcher and applier
        finder = qe.editors.nnmodules.GenericGraphMatcher(dropout_pattern)
        applier = DropoutRemoverApplier(dropout_pattern)
        # link pattern, matcher, and applier into the rewriter
        super(DropoutRemover, self).__init__('DropoutRemover', dropout_pattern, finder, applier)

__all__ = [
    'DropoutRemover',
]
