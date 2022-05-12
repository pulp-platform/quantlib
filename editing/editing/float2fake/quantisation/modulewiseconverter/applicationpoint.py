import torch.fx as fx
from typing import NamedTuple

from quantlib.editing.editing.editors.base import ApplicationPoint


class PartitionId(int):  # don't user `typing`'s `NewType`: https://stackoverflow.com/a/65965177
    pass


class _NodeWithPartition(NamedTuple):
    node: fx.Node
    id_:  PartitionId


class NodeWithPartition(ApplicationPoint, _NodeWithPartition):
    pass
