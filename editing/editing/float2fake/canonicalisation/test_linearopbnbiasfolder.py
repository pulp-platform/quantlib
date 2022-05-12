import unittest
from collections import OrderedDict
import torch.nn as nn


_N_FEATURES = 1
_KERNEL_SIZE = 1


class Conv2dBN2d(nn.Sequential):

    def __init__(self, conv2d_has_bias: bool):

        modules = OrderedDict([
            ('conv2d', nn.Conv2d(in_channels=_N_FEATURES, out_channels=_N_FEATURES, kernel_size=_KERNEL_SIZE, bias=conv2d_has_bias)),
            ('bn2d',   nn.BatchNorm2d(num_features=_N_FEATURES)),
        ])

        super(Conv2dBN2d, self).__init__(modules)
