from enum import IntEnum
import torch

from typing import Tuple


class Kind(IntEnum):
    ZEROS    = 0
    ONES     = 1
    RANDN    = 2
    LINSPACE = 3


class BatchSize(IntEnum):
    SINGLE = 2 ** 0
    SMALL  = 2 ** 4
    LARGE  = 2 ** 8


class InputSize(IntEnum):
    SMALL  = 0
    NORMAL = 1
    LARGE  = 2


class LinearOrConv(IntEnum):
    LINEAR = 0
    CONV   = 1


class LinearSize(IntEnum):
    SMALL  = 2 ** 6
    NORMAL = 2 ** 9
    LARGE  = 2 ** 12


class Conv2dChannels(IntEnum):
    SMALL  = 2 ** 6
    NORMAL = 2 ** 8
    LARGE  = 2 ** 10


class Conv2dSpatialSize(IntEnum):
    SMALL  = 2 ** 5  # CIFAR-10
    NORMAL = 2 ** 7  # ~ ImageNet / 2
    LARGE  = 2 ** 9  # ~ ImageNet * 2


class TensorGenerator(object):

    def __init__(self,
                 device: torch.device,
                 kind: Kind):

        self._device = device
        self._kind   = kind

    @property
    def size(self):
        raise NotImplementedError

    def __next__(self) -> torch.Tensor:

        if self._kind == Kind.ZEROS:
            x = torch.zeros(*self.size)

        elif self._kind == Kind.ONES:
            x = torch.ones(*self.size)

        elif self._kind == Kind.RANDN:
            x = torch.randn(*self.size)

        elif self._kind == Kind.LINSPACE:
            from functools import reduce
            n = reduce(lambda x, y: x * y, self.size)
            x = torch.arange(0, n).reshape(*self.size)  # x.view(-1) returns the linearly spaced values in a one-dimensional array

        else:
            raise ValueError

        return x.to(device=self._device)


class LinearTensorGenerator(TensorGenerator):

    def __init__(self,
                 device: torch.device,
                 kind: Kind,
                 batch_size: int,
                 n_channels: int):

        super(LinearTensorGenerator, self).__init__(device, kind)
        self._batch_size = batch_size
        self._n_channels = n_channels

    @property
    def size(self) -> Tuple[int, int]:
        return self._batch_size, self._n_channels


class Conv2dTensorGenerator(TensorGenerator):

    def __init__(self,
                 device: torch.device,
                 kind: Kind,
                 batch_size: int,
                 n_channels: int,
                 spatial_size: int):  # I assume squared images

        super(Conv2dTensorGenerator, self).__init__(device, kind)

        self._batch_size   = batch_size
        self._n_channels   = n_channels
        self._spatial_size = spatial_size

    @property
    def size(self) -> Tuple[int, int, int, int]:
        return self._batch_size, self._n_channels, self._spatial_size, self._spatial_size
