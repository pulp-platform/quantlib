# 
# create_tensors.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from enum import IntEnum
from collections import namedtuple
import torch

from typing import Tuple
from typing import Union


class BatchSize(IntEnum):
    SINGLE = 2 ** 0
    SMALL  = 2 ** 4
    LARGE  = 2 ** 8


class InputSize(IntEnum):
    SMALL  = 0
    NORMAL = 1
    LARGE  = 2


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


class Kind(IntEnum):
    ZEROS    = 0
    ONES     = 1
    RANDN    = 2
    LINSPACE = 3


Range = namedtuple('Range', ['start', 'end'])


class TensorGenerator(object):

    def __init__(self,
                 device: torch.device,
                 kind:   Kind,
                 range_: Union[Range, None] = None):

        self._device = device
        self._kind   = kind
        self._range  = range_

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
            assert self._range is not None
            from functools import reduce
            n = reduce(lambda x, y: x * y, self.size)
            x = torch.linspace(self._range.start, self._range.end, n).reshape(*self.size)  # x.view(-1) returns the linearly spaced values in a one-dimensional array

        else:
            raise ValueError

        return x.to(device=self._device)


class LinearTensorGenerator(TensorGenerator):

    def __init__(self,
                 device:     torch.device,
                 batch_size: int,
                 n_channels: int,
                 kind:       Kind,
                 range_:     Union[Range, None] = None):

        super(LinearTensorGenerator, self).__init__(device, kind, range_)
        self._batch_size = batch_size
        self._n_channels = n_channels

    @property
    def size(self) -> Tuple[int, int]:
        return self._batch_size, self._n_channels


class Conv2dTensorGenerator(TensorGenerator):

    def __init__(self,
                 device:       torch.device,
                 batch_size:   int,
                 n_channels:   int,
                 spatial_size: int,  # I assume squared images, so one integer is sufficient
                 kind:         Kind,
                 range_:       Union[Range, None] = None):

        super(Conv2dTensorGenerator, self).__init__(device, kind, range_)

        self._batch_size   = batch_size
        self._n_channels   = n_channels
        self._spatial_size = spatial_size

    @property
    def size(self) -> Tuple[int, int, int, int]:
        return self._batch_size, self._n_channels, self._spatial_size, self._spatial_size

