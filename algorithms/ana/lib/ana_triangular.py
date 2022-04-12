# 
# ana_triangular.py
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

from .ana_forward import forward_expectation, forward_mode, forward_random


# Wikipedia: https://en.wikipedia.org/wiki/Triangular_distribution


def forward(x_in, q, t, mi, sigma, strategy, training):

    # shift points with respect to the distribution's mean
    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    shifted_x_minus_t = x_in - mi - t.reshape(t_shape)

    # compute cumulative distribution function
    if training and sigma != 0.0:
        abs_shifted_x_minus_t_over_s = torch.abs(shifted_x_minus_t / sigma)
        cdf_temp = (torch.sign(shifted_x_minus_t) * abs_shifted_x_minus_t_over_s * (2 - abs_shifted_x_minus_t_over_s) + 1.0) / 2
        cdf = torch.where(abs_shifted_x_minus_t_over_s < 1.0, cdf_temp, (shifted_x_minus_t >= 0.0).float())
    else:
        cdf = (shifted_x_minus_t >= 0.0).float()

    # compute probability mass function over bins
    cdf = torch.vstack([torch.ones_like(cdf[0])[None, :], cdf, torch.zeros_like(cdf[-1][None, :])])
    pmf = cdf[:-1] - cdf[1:]

    # compute output
    if strategy == 0:  # expectation
        x_out = forward_expectation(pmf, q)
    elif strategy == 1:  # argmax sampling (i.e., mode)
        x_out = forward_mode(pmf, q)
    elif strategy == 2:  # random sampling
        x_out = forward_random(pmf, q)
    else:
        raise ValueError  # undefined strategy

    return x_out


def backward(grad_in, x_in, q, t, mi, sigma):

    # shift points with respect to the distribution's mean
    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    shifted_x_minus_t = x_in - mi - t.reshape(t_shape)

    # compute probability density function
    if sigma != 0.0:
        shifted_x_minus_t_over_s = torch.abs(shifted_x_minus_t / sigma)
        pdf = torch.max(torch.Tensor([0.0]).to(device=grad_in.device), 1.0 - shifted_x_minus_t_over_s) / sigma
    else:
        pdf = torch.zeros_like(shifted_x_minus_t)

    # compute gradient
    d = q[1:] - q[:-1]
    local_jacobian = torch.sum(d.reshape(t_shape) * pdf, dim=0)
    grad_out = grad_in * local_jacobian

    return grad_out
