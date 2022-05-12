"""The object guiding ``ActivationFinder``s when filtering ``fx.Node``s, and
``ActivationReplacer``s when building modular activations.
"""

import torch
import torch.nn as nn
from typing import NamedTuple, Tuple, Callable, Type


ActivationTarget = Callable[[torch.Tensor], torch.Tensor]


class NonModularTargets(NamedTuple):
    """Collections of ``fx.Node`` targets representing invocations of
    activation functions based on PyTorch's non-modular API.
    """
    inplace:    Tuple[ActivationTarget, ...]  # should the replacement be called inplace?
    noninplace: Tuple[ActivationTarget, ...]  # should the replacement not be called inplace?

    def __contains__(self, item):
        return (item in self.inplace) or (item in self.noninplace)


class ActivationSpecification(NamedTuple):
    """Container attaching the PyTorch modular API for an activation function
    to its non-modular counterparts.
    """
    module_class: Type[nn.Module]
    targets:      NonModularTargets
