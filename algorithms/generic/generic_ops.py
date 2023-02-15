from torch import nn

__all__ = [
    "CausalConv1d",
    "Multiply"
]

# used for canonicalization: calls to torch.mul, tensor.mul
class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x, y):
        return x*y


class CausalConv1d(nn.Conv1d):
    def __init__(self,
             in_channels,
             out_channels,
             kernel_size,
             stride=1,
             dilation=1,
             groups=1,
                 bias=True,
                 padding_mode='zeros'):

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 1, "Invalid Kernel Size in CausalConv1d: {}".format(kernel_size)
            k = kernel_size[0]
        else:
            k = kernel_size
        if isinstance(dilation, tuple):
            assert len(dilation) == 1, "Invalid Dilation in CausalConv1d: {}".format(dilation)
            dil = dilation[0]
        else:
            dil = dilation
        
        self.__padding = (k - 1) * dil

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dil,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
    def forward(self, input):
        pad_mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
        x = nn.functional.pad(input, (self.__padding, 0), mode=pad_mode)
        result = super(CausalConv1d, self).forward(x)
        return result
