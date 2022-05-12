"""The object passed between ``EpsTunnelInserterFinder``s and
``EpsTunnelInserterApplier``s."""

import torch.fx as fx
from typing import NamedTuple

from quantlib.editing.editing.editors.base import ApplicationPoint


# Since `ApplicationPoint` is an `ABC` object and `ABC`'s meta-class is
# `ABCMeta`, and `NamedTuple` is itself a meta-class but different from
# `ABCMeta`, we first need to instantiate a class from `NamedTuple`...

class _EpsTunnelNode(NamedTuple):
    node: fx.Node


# ... and then create our `ApplicationPoint` type.
class EpsTunnelNode(ApplicationPoint, _EpsTunnelNode):
    pass


# TODO: these abstractions are functionally identical to the corrseponding
#       ones for `ActivationModulariser`. Maybe we can find a way to create a
#       common base class?
