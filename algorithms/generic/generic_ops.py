from torch import nn

__all__ = [
    "Multiply"
]

# used for canonicalization: calls to torch.mul, tensor.mul
class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x, y):
        return x*y
