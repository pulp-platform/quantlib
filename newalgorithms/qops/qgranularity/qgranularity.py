import torch
from functools import reduce, partial
from operator import mul
from enum import Enum, auto, unique
import inspect
from typing import Tuple

from quantlib.newalgorithms.qbase.observer import MinMaxMeanVarObserver
from quantlib.newutils import quantlib_err_header


@unique
class QGranularity(Enum):
    PER_LAYER = auto()
    PER_CHANNEL = auto()


PER_CHANNEL_NON_BROADCASTABLE_DIMS = (0,)


def resolve_qgranularityspec(qgranularityspec: str) -> QGranularity:

    qgranularityspec = qgranularityspec.upper()

    try:
        qgranularity = QGranularity[qgranularityspec]
    except KeyError:
        raise ValueError(quantlib_err_header() + f"Unsupported QGranularity value: {qgranularityspec}.")

    return qgranularity


def make_broadcastable(x: torch.Tensor,
                       t: torch.Tensor,
                       non_broadcast_dims: Tuple[int]) -> torch.Tensor:

    up_shape = tuple(t.shape[i] for i in non_broadcast_dims)
    up_numel = reduce(mul, up_shape)

    # reshape ignoring the broadcasting dimensions (they will be filled by ones)
    if x.numel() == 1:
        x = torch.tile(x, up_shape)
    else:
        if x.numel() == up_numel:
            x = x.reshape(up_shape)
        else:
            raise RuntimeError(quantlib_err_header(obj_name=inspect.currentframe().f_code.co_name) + f"can not reshape `torch.Tensor` {x} with {x.numel()} elements to shape {up_shape}.")

    # fill the broadcasting dimensions with ones
    broadcast_shape = tuple(d if i in non_broadcast_dims else 1 for i, d in enumerate(t.shape))
    x = x.reshape(broadcast_shape)

    return x


make_broadcastable_per_channel = partial(make_broadcastable, non_broadcast_dims=PER_CHANNEL_NON_BROADCASTABLE_DIMS)

# is self._qgranularity is QGranularity.PER_CHANNEL:
#     self.zero.data = make_broadcastable_per_channel(self.zero.data, self.weight)

class PerLayerObserver(MinMaxMeanVarObserver):

    def __init__(self):
        super().__init__(subpopulation_dims=None)


class PerChannelObserver(MinMaxMeanVarObserver):

    def __init__(self):
        super().__init__(subpopulation_dims=PER_CHANNEL_NON_BROADCASTABLE_DIMS)
