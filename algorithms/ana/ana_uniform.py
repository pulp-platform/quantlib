# 
# ana_uniform.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

import ana_forward


def forward(x_in, q, t, mu, sigma, strategy, training):

    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    x_minus_t = x_in - mu - t.reshape(t_shape)

    if training and sigma != 0.:
        s_inv = 1 / sigma
        cdf = torch.clamp(0.5 * (x_minus_t * s_inv + 1), 0., 1.)
    else:
        cdf = (x_minus_t >= 0.).float()

    cdf = torch.vstack([torch.ones_like(cdf[0])[None, :], cdf, torch.zeros_like(cdf[-1][None, :])])
    pmf = cdf[:-1] - cdf[1:]

    if strategy == 0:  # expectation
        x_out = ana_forward.forward_expectation(pmf, q)
    elif strategy == 1:  # argmax sampling (i.e., mode)
        x_out = ana_forward.forward_mode(pmf, q)
    elif strategy == 2:  # random sampling
        x_out = ana_forward.forward_random(pmf, q)
    else:
        raise ValueError  # undefined strategy

    return x_out


def backward(grad_in, x_in, q, t, bmu, bsigma):

    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    x_minus_t = x_in - t.reshape(t_shape) - bmu

    if bsigma != 0.:
        s_inv = 1 / bsigma
        pdf = (torch.abs_(x_minus_t) <= bsigma).float() * (0.5 * s_inv)
    else:
        pdf = torch.zeros_like(grad_in)

    d = q[1:] - q[:-1]
    local_jacobian = torch.sum(d.reshape(t_shape) * pdf, 0)
    grad_out = grad_in * local_jacobian

    return grad_out

