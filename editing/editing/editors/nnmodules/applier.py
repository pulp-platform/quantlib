import torch.fx as fx

from ..base import Applier
from .applicationpoint import NodesMap
from .pattern.base import NNModulePattern


class NNModuleApplier(Applier):

    def __init__(self, pattern: NNModulePattern):
        super(NNModuleApplier, self).__init__()
        self._pattern = pattern

    @property
    def pattern(self) -> NNModulePattern:
        return self._pattern

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        raise NotImplementedError
