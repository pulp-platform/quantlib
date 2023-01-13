# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
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
