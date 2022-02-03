import torch
from typing import Tuple

from ..qrange import UNKNOWN, QRange


# aliases (for readability)
UNSPECIFIED = torch.Tensor([float('nan')])  # `ndim == 1` and `shape == (1,)`


def init_qhparams(qrange: QRange) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialise the hyper-parameters describing a quantiser."""

    zero     = torch.Tensor([qrange.offset]) if qrange.offset is not UNKNOWN else UNSPECIFIED
    n_levels = torch.Tensor([qrange.n_levels])
    step     = torch.Tensor([qrange.step])
    scale    = UNSPECIFIED

    return zero, n_levels, step, scale
