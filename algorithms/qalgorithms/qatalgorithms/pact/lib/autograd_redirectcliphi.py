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
from typing import Tuple

from quantlib.algorithms.qbase.qrange import IMPLICIT_STEP


class _PACTRedirectClipHiGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, clip_lo: torch.Tensor, n_levels: torch.Tensor, step: torch.Tensor) -> torch.Tensor:

        multiplier = -(n_levels - 2) / n_levels
        multiplier[(n_levels % 2 != 0) | (step != IMPLICIT_STEP)] = -1.0

        ctx.save_for_backward(multiplier)

        clip_hi = multiplier * clip_lo
        return clip_hi

    @staticmethod
    def backward(ctx, g_in: torch.Tensor) -> Tuple[torch.Tensor, None, None]:

        multiplier, = ctx.saved_tensors

        g_out = multiplier * g_in
        return g_out, None, None
