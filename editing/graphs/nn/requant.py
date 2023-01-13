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

class RequantClipFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lo : float,
        hi : float
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clip(min=lo, max=hi)

    @staticmethod
    def symbolic(
        g:  torch.Graph,
        x:  torch.Value,
        lo: float,
        hi: float
    ) -> torch.Value:

        return g.op(
            "Clip",
            x,
            min_f=lo,
            max_f=hi
        )

class Requantisation(nn.Module):

    def __init__(self,
                 mul:      torch.Tensor,
                 add:      torch.Tensor,
                 zero:     torch.Tensor,
                 n_levels: torch.Tensor,
                 D:        torch.Tensor = torch.Tensor([2 ** 24])):

        super(Requantisation, self).__init__()

        self.register_buffer('mul',      mul)
        self.register_buffer('add',      add)
        self.register_buffer('div',      D)
        self.register_buffer('zero',     zero)
        self.register_buffer('n_levels', n_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x * self.mul
        x = x + self.add
        x = torch.floor(x / self.div)  # This operation can be implemented in integer digital arithmetic as a right-shift by :math:`\log_{2}(D)` places; divisions can be avoided.
        lo = float(self.zero)
        hi = float(self.zero + self.n_levels - 1)
        x = RequantClipFn.apply(x, lo, hi)

        return x
