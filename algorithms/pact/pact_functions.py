# pact_functions.py
# Francesco Conti <f.conti@unibo.it>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
#
# Copyright (C) 2018-2021 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

__all__ = ['PACT_QuantFunc', 'AlmostSymmQuantFunc']

# PACT activation: https://arxiv.org/pdf/1805.06085.pdf
class PACT_QuantFunc(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function (asymmetric), using a floor function.

        Implements a :py:class:`torch.autograd.Function` for quantizing weights in :math:`Q` bits using an asymmetric PACT-like strategy (original
        PACT is applied only to activations, using DoReFa-style weights).
        In forward propagation, the function is defined as

        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [-\alpha,+\beta) } (\mathbf{x})\right\rfloor \cdot \varepsilon

        where :math:`\varepsilon` is the quantization precision:

        .. math::
            \varepsilon = (\alpha+\beta) / (2^Q - 1)

        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as

        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}

        It can be applied by using its static `.apply` method:

    :param input: the tensor containing :math:`x`, the weights to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param clip_lo: the value of the lower clipping bounds - either a unique value or a per-channel tensor
    :type  clip_lo: `torch.Tensor` or float
    :param clip_hi: the value of the upper clipping bounds
    :type  clip_hi: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default unused, 0).
    :type  delta: `torch.Tensor` or float
    :param clip_gradient: if True, zero-out gradients outside of the clipping range.
    :type  clip_gradient: bool

    :return: The quantized weights tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, clip_lo, clip_hi, delta=0, clip_gradient=False):
        # we quantize also clip_lo, beta. for beta it's "cosmetic", for clip_lo it is
        # substantial, because also clip_lo will be represented as a wholly integer number
        # down the line
        clip_lo_quant = (clip_lo.item() / (eps+delta)).floor() * eps
        clip_hi_quant  = (clip_hi.item() / (eps+delta)).floor() * eps
        where_input_nonclipped = (input >= clip_lo_quant) * (input < clip_hi_quant)
        where_input_lo = (input < clip_lo_quant)
        where_input_hi = (input >= clip_hi_quant)
        ctx.save_for_backward(where_input_nonclipped, where_input_lo, where_input_hi, clip_gradient)
        return ((input.clamp(clip_lo_quant.item(), clip_hi_quant.item()) / (eps+delta)).floor()) * eps

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_lo, where_input_hi, clip_gradient = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        if clip_gradient:
            grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        else:
            grad_input = grad_output
        grad_upper = torch.where(where_input_hi, grad_output, zero).sum().expand(1)
        # beta is the lower bound; making it larger will make the output smaller
        grad_lower  = torch.where(where_input_lo, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_lower, grad_upper

# to
class AlmostSymmQuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, clip_lo, n_levels):
        torch._assert(clip_lo < 0, "Big problem: lower_bound passed to AlmostSymmQuantFunc is not negative!!! Everything will break!")
        if n_levels % 2 == 0:
            clip_hi = -clip_lo * (n_levels-2)/n_levels
        else:
            clip_hi = -clip_lo
        ctx.save_for_backward(n_levels)
        return clip_hi

    @staticmethod
    def backward(ctx, grad_output):
        n_levels = ctx.saved_variables
        if n_levels % 2 == 0:
            grad_lo = -grad_output * (n_levels-2)/n_levels
        else:
            grad_lo = -grad_output
        return grad_lo, None
