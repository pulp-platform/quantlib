# 
# ana_normal.py
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
from scipy.stats import norm

from .ana_forward import forward_expectation, forward_mode, forward_random


# Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution


def forward(x_in, q, t, mi, sigma, strategy, training):

    is_cuda = x_in.is_cuda  # if one ``torch.Tensor`` operand is on GPU, all should be

    # shift points with respect to the distribution's mean
    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    shifted_x_minus_t = x_in - mi - t.reshape(t_shape)

    # compute cumulative distribution function
    if training and sigma != 0.0:
        if is_cuda:
            shifted_x_minus_t = shifted_x_minus_t.cpu()
            sigma    = sigma.cpu()
        cdf = torch.from_numpy(norm.cdf(shifted_x_minus_t.numpy(), 0.0, sigma.numpy())).to(dtype=x_in.dtype)
        if is_cuda:
            cdf = cdf.to(device=x_in.device)
    else:
        cdf = (shifted_x_minus_t >= 0.0).float()
        cdf = cdf.to(dtype=x_in.dtype)

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

    is_cuda = grad_in.is_cuda  # if one ``torch.Tensor`` operand is on GPU, all should be

    # shift points with respect to the distribution's mean
    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    shifted_x_minus_t = x_in - mi - t.reshape(t_shape)

    # compute probability density function
    if sigma != 0.0:
        if is_cuda:
            shifted_x_minus_t = shifted_x_minus_t.cpu()
            sigma    = sigma.cpu()
        pdf = torch.from_numpy(norm.pdf(shifted_x_minus_t.numpy(), 0.0, sigma.numpy())).to(dtype=grad_in.dtype)
        if is_cuda:
            pdf = pdf.to(device=grad_in.device)
    else:
        pdf = torch.zeros_like(shifted_x_minus_t)

    # compute gradient
    d = q[1:] - q[:-1]
    local_jacobian = torch.sum(d.reshape(t_shape) * pdf, dim=0)
    grad_out = grad_in * local_jacobian

    return grad_out
