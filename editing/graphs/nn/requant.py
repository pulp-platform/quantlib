import torch
import torch.nn as nn

class RequantClipFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lo : float,
        hi : float
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clip(min=lo, max=hi)

    @staticmethod
    def symbolic(
        g:  torch.Graph,
        x:  torch.Value,
        lo: float,
        hi: float
    ) -> torch.Value:

        return g.op(
            "Clip",
            x,
            min_f=lo,
            max_f=hi
        )

class Requantisation(nn.Module):

    def __init__(self,
                 mul:      torch.Tensor,
                 add:      torch.Tensor,
                 zero:     torch.Tensor,
                 n_levels: torch.Tensor,
                 D:        torch.Tensor = torch.Tensor([2 ** 24])):

        super(Requantisation, self).__init__()

        self.register_buffer('mul',      mul)
        self.register_buffer('add',      add)
        self.register_buffer('div',      D)
        self.register_buffer('zero',     zero)
        self.register_buffer('n_levels', n_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x * self.mul
        x = x + self.add
        x = torch.floor(x / self.div)  # This operation can be implemented in integer digital arithmetic as a right-shift by :math:`\log_{2}(D)` places; divisions can be avoided.
        lo = float(self.zero)
        hi = float(self.zero + self.n_levels - 1)
        x = RequantClipFn.apply(x, lo, hi)

        return x
