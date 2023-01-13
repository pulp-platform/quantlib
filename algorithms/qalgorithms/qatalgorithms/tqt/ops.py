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

import torch
import torch.nn as nn
from typing import Tuple, Dict, Union


from quantlib.newquant import resolve_quantspec
from quantlib.algorithms.qbase import QATActivation, QATLinear


class TQTActivation(QATActivation):

    def __init__(self,
                 quantspec: Union[Tuple[int, ...], Dict[str, int], str],
                 pinrange: bool,
                 unquant_module: Union[None, nn.Module] = None):

        qrange = resolve_quantspec(quantspec)

        self._n_levels = qrange.n_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the TQT operation to the input array."""
        pass
