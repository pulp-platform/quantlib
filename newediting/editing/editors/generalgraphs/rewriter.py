import torch.fx as fx
from typing import List

from .pattern import GenericGraphPattern
from .matcher import GenericGraphMatcher
from quantlib.newediting.editing.editors.base import ApplicationPoint, Rewriter


class GenericGraphRewriter(Rewriter):

    def __init__(self,
                 pattern: GenericGraphPattern,
                 name: str):

        super(GenericGraphRewriter, self).__init__(name)
        self._matcher = GenericGraphMatcher(pattern=pattern)

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        matches = self._matcher.find(g)
        return [ApplicationPoint(rewriter=self, graph=g, core=match) for match in matches]

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        raise NotImplementedError

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:
        raise NotImplementedError
