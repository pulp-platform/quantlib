"""The object passed between ``ActivationFinder``s and ``ActivationReplacer``s."""

import torch.fx as fx
from typing import NamedTuple

from quantlib.editing.editing.editors.base import ApplicationPoint


# Since `ApplicationPoint` is an `ABC` object and `ABC`'s meta-class is
# `ABCMeta`, and `NamedTuple` is itself a meta-class but different from
# `ABCMeta`, we first need to instantiate a class from `NamedTuple`...

class _ActivationNode(NamedTuple):
    node: fx.Node


# ... and then create our `ApplicationPoint` type.
class ActivationNode(ApplicationPoint, _ActivationNode):
    pass
