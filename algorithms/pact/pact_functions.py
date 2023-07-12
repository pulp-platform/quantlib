#
# pact_functions.py
#
# Author(s):
# Francesco Conti <f.conti@unibo.it>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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


__all__ = [
    'PACTQuantFunc',
    'AlmostSymmQuantFunc',
    'PACTQuantize',
    'TQTQuantFunc',
    'TQTQuantize'
]


#PACT activation: https://arxiv.org/pdf/1805.06085.pdf

class PACTQuantFunc(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function (asymmetric), using a floor or round function.

        Implements a :py:class:`torch.autograd.Function` for quantizing weights in :math:`Q` bits using an asymmetric PACT-like strategy (original
        PACT is applied only to activations, using DoReFa-style weights).
        In forward propagation, the function is defined as

        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [\alpha, \beta) } (\mathbf{x})\right\rfloor \cdot \varepsilon

        where :math:`\varepsilon` is the quantization precision:

        .. math::
            \varepsilon = (\beta-\alpha) / (2^Q - 1)

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
    :param floor:    If True, perform flooring on to get integer representation. if False, perform rounding.
    :param clip_gradient: if True, zero-out gradients outside of the clipping range.
    :type  clip_gradient: bool
    :param noisy: add half uniform noise with max. magnitude of half an MSB to the output. not recommended to use, it doesn't help...

    :return: The quantized tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, clip_lo, clip_hi, floor=True, clip_gradient=True, noisy=False):
        where_input_nonclipped = torch.mul(input >= clip_lo, input < clip_hi)
        where_input_lo = (input < clip_lo)
        where_input_hi = (input >= clip_hi)
        ctx.save_for_backward(where_input_nonclipped, where_input_lo, where_input_hi, clip_gradient, clip_lo)
        # for completeness' sake (e.g. to reproduce the results from the
        # PACT+SAWB paper), we allow for outputs which are not a multiple of
        # eps.
        # to ensure hardware compatibility, it is the downstream user's
        # responsibility to ensure that clip_lo/clip_hi are multiples of eps!
        input_unrounded_int = (input.clamp(clip_lo, clip_hi + 1e-7) - clip_lo)/ eps
        if noisy:
            noise = torch.rand(input_unrounded_int.size(), device=input_unrounded_int.device) - 0.5
            input_unrounded_int += noise

        # for weights, we want to use rounding - for activations, we will round
        # in hardware so represent this here too
        input_rounded_int = input_unrounded_int.floor() if floor else input_unrounded_int.round()
        return input_rounded_int * eps + clip_lo

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_lo, where_input_hi, clip_gradient, clip_lo = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        if clip_gradient:
            grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        else:
            grad_input = grad_output
        reduce_dims = tuple(range(len(grad_output.shape)))
        if len(clip_lo.shape) > 1:
            # this works only for weights due to activations' batch dimensions,
            # but we don't support per-channel quantization of activations so
            # it's OK
            reduce_dims = reduce_dims[1:]

        grad_upper = torch.where(where_input_hi, grad_output, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        # clip_lo is the lower bound; making it larger will make the output larger
        # if input was clipped. the gradient propagation is thus identical for
        # lower and upper bounds!
        grad_lower  = torch.where(where_input_lo, grad_output, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        return grad_input, None, grad_lower, grad_upper, None, None, None





# a wrapper for PACTQuantFunc to allow kwargs
def PACTQuantize(x, eps, clip_lo, clip_hi, floor=True, clip_gradient=True, noisy=False):
    return PACTQuantFunc.apply(x, eps, clip_lo, clip_hi, floor, clip_gradient, noisy)

class TQTQuantFunc(torch.autograd.Function):
    r"""TQT (Trained Quantization Thresholds) quantization function (asymmetric).

        Implements a :py:class:`torch.autograd.Function` for quantizing weights or activations in :math:`Q` bits using the technique detailed in
        the TQT paper (https://arxiv.org/abs/1903.08066)
        In forward propagation, the function is defined as

        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lround\mathrm{clip}_{ [\alpha, \beta) } (\mathbf{x})\right\rround \cdot \varepsilon

        where :math:`\varepsilon` is the quantization precision:

        .. math::
            \varepsilon = (-\alpha+\beta) / (2^Q - 1)


        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as

        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}
        Along with the input to the quantizer, a gradient is propagated to the log2 of the thresholding parameter :math:`t`, which is assumed to determine the support of the clipping
        range logarithmically, i.e. :math: `\beta - \alpha \propto 2^{\log_2 t}`

        It can be applied by using its static `.apply` method:

    :param input: the tensor containing :math:`x`, the weights to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param clip_lo: the value of the lower clipping bounds - either a unique value or a per-channel tensor
    :type  clip_lo: `torch.Tensor` or float
    :param clip_hi: the value of the upper clipping bounds
    :type  clip_hi: `torch.Tensor` or float
    :param floor:    If True, perform flooring on to get integer representation. if False, perform rounding.
    :param clip_gradient: if True, zero-out gradients outside of the clipping range.
    :type  clip_gradient: bool

    :return: The quantized tensor.
    :rtype:  `torch.Tensor`

    """

    @staticmethod
    def forward(ctx, input, eps, log_t, clip_lo, clip_hi, beta, running_grad_var, running_beta, clip_grad_logt, rounding):
        where_input_nonclipped = torch.mul((input >= clip_lo), (input < clip_hi))
        where_input_lo = (input < clip_lo)
        where_input_hi = (input >= clip_hi)

        # for completeness' sake (e.g. to reproduce the results from the
        # PACT+SAWB paper), we allow for outputs which are not a multiple of
        # eps.
        # to ensure hardware compatibility, it is the downstream user's
        # responsibility to ensure that clip_lo/clip_hi are multiples of eps!
        input_unrounded_int = (input.clamp(clip_lo, clip_hi) - clip_lo)/ eps

        # for weights, we want to use rounding - for activations, we will round
        # in hardware so represent this here too
        if rounding:
            input_rounded_int = input_unrounded_int.round()
        else:
            input_rounded_int = input_unrounded_int.floor()
        input_quant = input_rounded_int * eps + clip_lo
        quant_error = input_quant - input
        ctx.save_for_backward(where_input_nonclipped, where_input_lo, where_input_hi, clip_lo, clip_hi, quant_error, beta, running_grad_var, running_beta, clip_grad_logt)
        return input_quant

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_nonclipped, where_input_lo, where_input_hi, clip_lo, clip_hi, quant_error, beta, running_grad_var, running_beta, clip_grad_logt = ctx.saved_variables
        zero = torch.zeros(1).to(where_input_nonclipped.device)
        grad_input = torch.where(where_input_nonclipped, grad_output, zero)

        reduce_dims = tuple(range(len(grad_output.shape)))
        if len(clip_lo.shape) > 1:
            # this works only for weights due to activations' batch dimensions,
            # but we don't support per-channel quantization of activations so
            # it's OK
            reduce_dims = reduce_dims[1:]
        ln2 = torch.log(torch.tensor(2.).to(clip_lo.device))

        # clipped high
        grad_logt =  clip_hi * torch.where(where_input_hi, grad_output, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        # clipped low
        grad_logt = grad_logt + clip_lo * torch.where(where_input_lo, grad_output, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        # unclipped
        grad_logt  = grad_logt + torch.where(where_input_nonclipped, quant_error * grad_output, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        # scale by log2
        grad_logt = grad_logt * ln2
        # normalize and bias-correct (see appendix B in paper) the gradient
        grad_var = beta * running_grad_var + (1-beta) * grad_logt**2
        #running_grad_var.copy_(grad_var.reshape(running_grad_var.shape))
        # we have to hack these with .data in order to bamboozle the
        # autograd version checker!
        running_grad_var.data.copy_(grad_var.reshape(running_grad_var.shape))
        running_beta.data.mul_(beta)
        grad_var /= (1 - running_beta)
        grad_logt /= (torch.sqrt(grad_var) + 1e-5)
        if clip_grad_logt:
            grad_logt = torch.tanh(grad_logt)

        return grad_input, None, grad_logt, None, None, None, None, None, None, None

#wrapper to allow kwargs
def TQTQuantize(input, eps, log_t, clip_lo, clip_hi, beta, running_grad_var, running_beta, clip_grad_logt, rounding=True):
    return TQTQuantFunc.apply(input, eps, log_t, clip_lo, clip_hi, beta, running_grad_var, running_beta, clip_grad_logt, rounding)

class AlmostSymmQuantFunc(torch.autograd.Function):
    r"""Helper functional which returns an upper clipping bound which is
    'quasy-symmetrical' to the :math: `clip_{lo}` provided. The quantization levels
    generated from the clipping bounds :math: `(clip_{lo}, clip_{hi})` will map to an
    integer range with just multiplicative scaling (no zero-point offset) if
    n_levels is a power of 2.
    :math: `clip_{lo}` must be :math: `<0`!
    """

    @staticmethod
    def forward(ctx, clip_lo, n_levels):
        r"""
        Returns the 'quasi-symmetric' upper clipping bound :math: `clip_{hi}` to the supplied :math: `clip_{lo}`:

        :math: `clip_{hi} = -clip_{lo} * \frac{n_{levels}/2 - 1}{n_levels/2}` if n_levels is even (e.g., 127 if clip_lo is -128)
        :math: `clip_{hi} = -clip_{lo}` if n_levels is odd
        """

        torch._assert(torch.all(clip_lo <= 0), "Big problem: `clip_lo` passed to AlmostSymmQuantFunc is not negative: everything will break!")
        if n_levels == 2:
            scale = 1.
            ctx.save_for_backward(scale)
            return -clip_lo
        if n_levels % 2 == 0:
            scale = torch.tensor(-(n_levels-2)/n_levels, device=clip_lo.device)
        else:
            scale = torch.tensor(-1., device=clip_lo.device)

        clip_hi = scale * clip_lo
        ctx.save_for_backward(scale)
        return clip_hi

    @staticmethod
    def backward(ctx, grad_output):
        scale,  = ctx.saved_variables
        grad_lo = scale * grad_output
        return grad_lo, None
