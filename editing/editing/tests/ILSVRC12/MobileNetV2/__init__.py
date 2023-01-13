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

from .preprocess import ILSVRC12MNv2Transform
from .mobilenetv2 import MobileNetV2
from .headrewriter import MNv2HeadRewriter

import os
from typing import List, Union


def MNv2_checkpoints() -> List[Union[os.PathLike, str]]:

    path_package = os.path.dirname(os.path.realpath(__file__))
    path_checkpoints = os.path.join(path_package, 'checkpoints')

    if not os.path.isdir(path_checkpoints):
        raise FileNotFoundError

    checkpoints = [os.path.join(path_checkpoints, filename) for filename in os.listdir(path_checkpoints) if (filename.endswith('.ckpt'))]

    return checkpoints
