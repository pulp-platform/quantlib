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

import operator
import torch

from quantlib.editing.editing.editors.optrees import OpSpec, OpTreeFinder
from quantlib.editing.graphs.fx import FXOpcodeClasses


addspec = OpSpec([
    (next(iter(FXOpcodeClasses.CALL_FUNCTION.value)), (operator.add, torch.add,)),
    (next(iter(FXOpcodeClasses.CALL_METHOD.value)),   ('add',)),
])


class AddTreeFinder(OpTreeFinder):
    def __init__(self):
        super(AddTreeFinder, self).__init__(addspec)
