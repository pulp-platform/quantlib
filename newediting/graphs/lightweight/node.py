from __future__ import annotations
import torch.nn as nn
from typing import NamedTuple, List


class Node(NamedTuple):

    name:   str
    module: nn.Module

    @property
    def path(self) -> List[str]:
        return self.name.split('.')

    def __eq__(self, other: Node) -> bool:
        return isinstance(other, Node) and (self.name == other.name) and (self.module is other.module)
