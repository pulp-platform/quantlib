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

from collections import OrderedDict
import torch.nn as nn
from typing import Type

from ...qmodules.qmodules.qmodules import _QModule
from ...qmodules import SUPPORTED_FPMODULES


class ModuleMapping(OrderedDict):
    """Map floating-point ``nn.Module``s to their fake-quantised counterparts.

    QuantLib developers who implement new PTQ/QAT algorithms should define a
    ``ModuleMapping`` object inside each algorithm-specific sub-package, then
    add it to the global register of PTQ/QAT algorithms.

    """

    def __setitem__(self, fpmodule: Type[nn.Module], fqmodule: Type[_QModule]):

        if not isinstance(fpmodule, type(nn.Module)):
            raise TypeError  # not a floating-point module
        if not isinstance(fqmodule, type(_QModule)):
            raise TypeError  # not a fake-quantised module

        if not (fpmodule in SUPPORTED_FPMODULES):
            raise ValueError  # QuantLib does not support a fake-quantised counterpart for this `nn.Module` class

        super(ModuleMapping, self).__setitem__(fpmodule, fqmodule)
