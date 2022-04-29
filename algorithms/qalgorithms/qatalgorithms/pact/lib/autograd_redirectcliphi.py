import torch
from typing import Tuple

from quantlib.algorithms.qbase.qrange import IMPLICIT_STEP


class _PACTRedirectClipHiGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, clip_lo: torch.Tensor, n_levels: torch.Tensor, step: torch.Tensor) -> torch.Tensor:

        multiplier = -(n_levels - 2) / n_levels
        multiplier[(n_levels % 2 != 0) | (step != IMPLICIT_STEP)] = -1.0

        ctx.save_for_backward(multiplier)

        clip_hi = multiplier * clip_lo
        return clip_hi

    @staticmethod
    def backward(ctx, g_in: torch.Tensor) -> Tuple[torch.Tensor, None, None]:

        multiplier, = ctx.saved_tensors

        g_out = multiplier * g_in
        return g_out, None, None
