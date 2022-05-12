"""The data structure used by ``ModuleWiseFinder``s to identify application
points.
"""

from collections import OrderedDict
import torch.nn as nn
from typing import Tuple


class NameToModule(OrderedDict):
    """A map from symbolic names to ``nn.Module`` objects."""

    @staticmethod
    def split_path_to_target(target: str) -> Tuple[str, str]:
        """Separate an ``nn.Module``'s name from its parent's qualified name.

        The hierarchy of ``nn.Module``s that composes a PyTorch network is
        captured by the module names through dotted notation (i.e., the names
        are *qualified*).
        """
        *ancestors, child = target.rsplit('.')
        path_to_parent = '.'.join(ancestors) if len(ancestors) > 0 else ''
        return path_to_parent, child

    def __setitem__(self, name: str, module: nn.Module):
        """We enforce type checking by overwriting the parent class's method."""

        if not isinstance(name, str):
            raise TypeError
        if not isinstance(module, nn.Module):
            raise TypeError

        super(NameToModule, self).__setitem__(name, module)
