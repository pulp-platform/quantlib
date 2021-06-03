#
# pact_functions.py
# Francesco Conti <f.conti@unibo.it>
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
    :param alpha: the value of :math:`\alpha`.
    :type  alpha: `torch.Tensor` or float
    :param beta: the value of :math:`\beta`.
    :type  beta: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default unused, 0).
    :type  delta: `torch.Tensor` or float
    :param clip_gradient: if True, zero-out gradients outside of the clipping range.
    :type  clip_gradient: bool
    
    :return: The quantized weights tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, alpha, beta, delta=0, clip_gradient=False):
        # we quantize also alpha, beta. for beta it's "cosmetic", for alpha it is 
        # substantial, because also alpha will be represented as a wholly integer number
        # down the line
        alpha_quant = (alpha.item() / (eps+delta)).ceil()  * eps
        beta_quant  = (beta.item()  / (eps+delta)).floor() * eps
        where_input_nonclipped = (input >= -alpha_quant) * (input < beta_quant)
        where_input_ltalpha = (input < -alpha_quant)
        where_input_gtbeta = (input >= beta_quant)
        ctx.save_for_backward(where_input_nonclipped, where_input_ltalpha, where_input_gtbeta, clip_gradient)
        return ((input.clamp(-alpha_quant.item(), beta_quant.item()) / (eps+delta)).floor()) * eps

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_ltalpha, where_input_gtbeta, clip_gradient = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        if clip_gradient:
            grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        else:
            grad_input = grad_output
        grad_alpha = torch.where(where_input_ltalpha, grad_output, zero).sum().expand(1)
        grad_beta  = torch.where(where_input_gtbeta, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_alpha, grad_beta
