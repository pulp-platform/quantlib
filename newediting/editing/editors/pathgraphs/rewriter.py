import torch.fx as fx
from typing import Callable, List

from .pattern import PathGraphDescription, PathGraphPattern
from .matcher import PathGraphMatcher
from ..base import ApplicationPoint, Rewriter


class PathGraphRewriter(Rewriter):

    def __init__(self,
                 name:              str,
                 pgd:               PathGraphDescription,
                 symbolic_trace_fn: Callable):

        super(PathGraphRewriter, self).__init__(name)

        pattern = PathGraphPattern(pgd, symbolic_trace_fn)
        self._matcher = PathGraphMatcher(pattern)

    @property
    def pattern(self) -> PathGraphPattern:
        return self._matcher.pattern

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        matches = self._matcher.find(g)
        return [ApplicationPoint(rewriter=self, graph=g, core=match.nodes_map) for match in matches]

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        raise NotImplementedError

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:
        raise NotImplementedError
