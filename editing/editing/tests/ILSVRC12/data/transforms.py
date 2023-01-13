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

from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose


ILSVRC12STATS = \
    {
        'normalize':
            {
                'mean': (0.485, 0.456, 0.406),
                'std':  (0.229, 0.224, 0.225)
            },
        'quantize':
            {
                'min': -2.1179039478,
                'max': 2.6400001049,
                'eps': 0.020625000819563866
            }
    }


class ILSVRC12Normalize(Normalize):
    def __init__(self):
        super(ILSVRC12Normalize, self).__init__(**ILSVRC12STATS['normalize'])


class ILSVRC12Transform(Compose):

    def __init__(self, image_size: int = 224):

        # validate arguments
        RESIZE_SIZE = 256
        if not (image_size <= RESIZE_SIZE):
            raise ValueError  # otherwise, `CenterCrop` can not yield the desired image

        transforms = [Resize(RESIZE_SIZE),
                      CenterCrop(image_size),
                      ToTensor(),
                      ILSVRC12Normalize()]

        super(ILSVRC12Transform, self).__init__(transforms)
