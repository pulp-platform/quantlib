# 
# ana_forward.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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


def forward_expectation(pmf: torch.Tensor, q: torch.Tensor):
    q_shape = [q.numel()] + [1 for _ in range(1, pmf.dim())]  # prepare for broadcasting
    return torch.sum(pmf * q.reshape(q_shape), dim=0)


def forward_mode(pmf: torch.Tensor, q: torch.Tensor):
    return q[torch.argmax(pmf, dim=0)]


def forward_random(pmf: torch.Tensor, q: torch.Tensor):
    q_shape = [q.numel()] + [1 for _ in range(1, pmf.dim())]  # prepare for broadcasting
    us = torch.rand_like(pmf[0]).unsqueeze(0)
    ge = torch.cumsum(pmf, dim=0) - us  # compute the position of the random numbers with respect to the segments
    sc = (ge[:-1] * ge[1:] <= 0).float()  # detect sign change
    idxs = torch.vstack([(1.0 - torch.sum(sc, dim=0)).unsqueeze(0), sc])  # the reduction performed by `torch.sum` must be compensated by the "inflation" provided by `unsqueeze`, so that the tensors can be concatenated
    return torch.sum(idxs * q.reshape(q_shape), dim=0)
