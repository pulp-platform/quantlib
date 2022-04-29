import torch.nn as nn
import torch.fx as fx
from typing import Set, Union

from ..base import NNModuleWithCheckersSpecType, resolve_nnmodulewithcheckersspec
from ..base import NNModulePattern, SymbolicTraceFnType


class NNSequentialPattern(NNModulePattern):

    def __init__(self,
                 nnmodulewithcheckersspec: NNModuleWithCheckersSpecType,
                 symbolic_trace_fn: SymbolicTraceFnType):

        # verify that the pattern is an `nn.Sequential` object
        module_with_checkers = resolve_nnmodulewithcheckersspec(nnmodulewithcheckersspec)
        if not isinstance(module_with_checkers.module, nn.Sequential):
            raise TypeError

        super(NNSequentialPattern, self).__init__(module_with_checkers, symbolic_trace_fn)
        self._leakable_nodes: Set[fx.Node] = set()

    @property
    def leakable_nodes(self) -> Set[fx.Node]:
        return self._leakable_nodes

    def _set_leakable_nodes(self, nodes: Union[fx.Node, Set[fx.Node]]) -> None:
        """It might be not relevant that some target nodes in the match are
        read by downstream ``fx.Node``s outside the match, since one of the
        following conditions will be satisfied:
          * the outputs of the target nodes will remain the same even after
            the rewriting;
          * all the users of the target nodes will read the same value after
            the rewriting.

        This function is intended to be called at most once from the
        constructor method of children classes, after calling the constructor
        of the parent (i.e., this) class; in this way, the sub-class developer
        can specify whether certain nodes are to be treated differently during
        the pattern matching performed by ``PathGraphMatcher``s.

        """

        # validate input type
        if not (isinstance(nodes, fx.Node) or (isinstance(nodes, set) and all(isinstance(n, fx.Node) for n in nodes))):
            raise TypeError

        # canonicalise input
        if isinstance(nodes, fx.Node):
            nodes = {nodes}

        # check input validity
        if not all(n in self.fxg.nodes for n in nodes):
            raise ValueError

        self._leakable_nodes = self._leakable_nodes.union(nodes)
