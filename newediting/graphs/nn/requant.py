import torch
import torch.nn as nn


class Requant(nn.Module):

    def __init__(self,
                 mul:      torch.Tensor,
                 add:      torch.Tensor,
                 zero:     torch.Tensor,
                 n_levels: torch.Tensor,
                 D:        torch.Tensor = torch.Tensor([2 ** 24])):

        super(Requant, self).__init__()

        self.register_buffer('mul', mul)
        self.register_buffer('add', add)
        self.register_buffer('div', D)
        self.zero     = zero
        self.n_levels = n_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x * self.mul
        x = x + self.add
        x = torch.floor(x / self.div)
        x = torch.clip(x, self.zero, self.zero + self.n_levels - 1)

        return x
