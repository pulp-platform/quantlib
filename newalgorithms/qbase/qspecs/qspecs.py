import torch
from enum import Enum, auto, unique
from functools import reduce
from operator import mul
import inspect
from typing import NewType
from typing import Tuple
from typing import Union, Optional

from ..qrange import QRange, UNKNOWN, ImplicitStep, IMPLICIT_STEP
from quantlib.newutils import quantlib_err_header


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
            raise RuntimeError(quantlib_err_header(inspect.currentframe().f_code.co_name) + f"can not reshape `torch.Tensor` {x} with {x.numel()} elements to shape {up_shape}.")

    # fill the broadcasting dimensions with ones
    broadcast_shape = tuple(d if i in non_broadcast_dims else 1 for i, d in enumerate(t.shape))
    x = x.reshape(broadcast_shape)

    return x


UnspecifiedZeroPoint = NewType('UnspecifiedZeroPoint', torch.Tensor)
UNSPECIFIED_ZEROPOINT = UnspecifiedZeroPoint(torch.Tensor(float('nan')).reshape(1))


UnspecifiedScale = NewType('UnspecifiedScale', torch.Tensor)
UNSPECIFIED_SCALE = UnspecifiedScale(torch.tensor(float('nan')).reshape(1))


@unique
class QGranularity(Enum):
    PER_LAYER = auto()
    PER_CHANNEL = auto()


NON_BROADCAST_DIM = (0,)


def init_qhparams(qrange: QRange,
                  granularity: QGranularity,
                  t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, ImplicitStep], Union[torch.Tensor, UnspecifiedScale]]:
    """Initialise the hyper-parameters describing a quantiser."""

    zero     = torch.tensor(qrange.offset).reshape(1) if qrange.offset is not UNKNOWN else UNSPECIFIED_ZEROPOINT
    n_levels = torch.tensor(qrange.n_levels).reshape(1)
    step     = torch.tensor(qrange.step).reshape(1) if qrange.step is not IMPLICIT_STEP else IMPLICIT_STEP
    scale    = UNSPECIFIED_SCALE

    if granularity == QGranularity.PER_CHANNEL:
        if t:  # use template tensor to resolve broadcasting
            zero     = make_broadcastable(zero,     t, NON_BROADCAST_DIM)
            n_levels = make_broadcastable(n_levels, t, NON_BROADCAST_DIM)
            step     = make_broadcastable(step,     t, NON_BROADCAST_DIM) if step is not IMPLICIT_STEP else IMPLICIT_STEP
            scale    = make_broadcastable(scale,    t, NON_BROADCAST_DIM)
        else:
            raise ValueError(quantlib_err_header(inspect.currentframe().f_code.co_name) + f"initialising per-channel quantisers requires an example `torch.Tensor` to resolve broadcasting, but received {type(t)}.")

    return zero, n_levels, step, scale
