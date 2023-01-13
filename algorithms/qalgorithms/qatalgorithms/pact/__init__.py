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

from quantlib.algorithms.qalgorithms import ModuleMapping

from .qactivations import NNMODULE_TO_PACTACTIVATION, PACTIdentity, PACTReLU, PACTReLU6, PACTLeakyReLU
from .qlinears     import NNMODULE_TO_PACTLINEAR, PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d

NNMODULE_TO_PACTMODULE = ModuleMapping(**NNMODULE_TO_PACTACTIVATION, **NNMODULE_TO_PACTLINEAR)

from .optimisers import PACTSGD, PACTAdam, PACTAdagrad
