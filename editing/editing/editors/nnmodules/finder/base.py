import torch.fx as fx
from typing import List

from ..applicationpoint import NodesMap
from ..pattern.base import NNModulePattern
from quantlib.editing.editing.editors.base import Finder


class NNModuleMatcher(Finder):
    """A class to find application points by pattern-matching."""

    def __init__(self, pattern: NNModulePattern):
        super(NNModuleMatcher, self).__init__()
        self._pattern = pattern

    @property
    def pattern(self) -> NNModulePattern:
        return self._pattern

    def find(self, g: fx.GraphModule) -> List[NodesMap]:
        raise NotImplementedError

    def check_aps_commutativity(self, aps: List[NodesMap]) -> bool:
        raise NotImplementedError
