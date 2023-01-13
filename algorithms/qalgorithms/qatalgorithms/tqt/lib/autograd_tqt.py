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


class TQTQuantiser(torch.autograd.Function):
    """TQT (Trained Quantization Thresholds) quantisation function.

    This function act component-wise on the input array. Its forward and
    backward passes implement the same functions as the PACT operation.
    However, there is a difference in that the learning signals that in PACT
    are directed to the lower and upper bounds of the clipping range
    :math:`[\alpha, \beta)` are replaced by a learning signal directed to a
    single parameter :math:`t`.

    In particular, :math:`t` is related to the clipping bounds :math:`alpha`
    and :math:`beta` in the following ways:

    * in TQT ``nn.Module``s using unsigned quantisers, the lower clipping
      bound :math:`\alpha \equiv 0` is pinned to zero, and :math:`t = \beta`
      is the value of the upper clipping bound;
    * in TQT ``nn.Module``s using signed quantisers, the bounds satisfy

      .. math::
         0 < \beta < |\alpha| \,,

       :math:`t = -\alpha` is the absolute value of the lower clipping bound.

    """

    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                eps: torch.Tensor,
                clip_lo: torch.Tensor,
                clip_hi: torch.Tensor,
                floor: bool,
                log_t: torch.Tensor,
                beta,  # TODO: annotate type
                beta_running,  # TODO: annotate type
                g_log_t_running_var,  # TODO: annotate type
                clip_g_log_t: bool,
                clip_g: bool = True) -> torch.Tensor:
        """Compute the forward pass of the TQT operation.

        Arguments:
            x: the array to be quantised.
            eps: the (precomputed) value of the quantum :math:`\varepsilon`.
            clip_lo: the lower clipping bound :math:`\alpha`.
            clip_hi: the upper clipping bound :math:`beta`.
            floor: whether to apply flooring (True) or rounding (False) to
                integerise the array.
            log_t: the logarithm (in base two) of the parameter :math:`t`;
                This argument is not used in the function, so what is its
                purpose? Due to PyTorch's autograd mechanics and the working
                of ``torch.autograd.Function`` classes, the fact that it
                appears in the signature of the ``forward`` function forces
                the programmer to return a corresponding gradient object in
                the corresponding ``backward`` call; the autograd engine will
                forward this object to the ``torch.Tensor`` storing the value
                of :math:`t`. It's a "hack".
            beta: the interpolation parameter of the running statistic. As an
                interpolation parameter, its value should be in the range
                :math:`[0, 1)`.
            beta_running: the decayed version of the ``beta`` argument, which
                makes the impact of the running variance stronger as training
                progresses. The larger the value of :math:`beta`, the slowlier
                this arithmetic progression will decrease. It is owned by the
                ``nn.Module`` calling this function, but it should not be
                modified outside of this function.
            g_log_t_running_var: the running variance of the gradients
                directed towards ``log_t``. It is owned by the ``nn.Module``
                calling this function, but it should not be modified outside
                of this function.
            clip_g_log_t: whether to "soft-clip" the gradient directed towards
                ``log_t`` using the hyperbolic tangent function (i.e., each
                component will be mapped to the range :math:`(-1, 1)`).
            clip_g: whether zeroing the components of the outgoing gradient
                array if the corresponding components of the input array are
                outside the clipping range :math:`[\alpha, \beta)`.

        """

        # partition the components of the input tensor with respect to the clipping bounds
        where_x_lo = (x < clip_lo)
        where_x_nc = (clip_lo <= x) * (x < clip_hi)  # non-clipped
        where_x_hi = (clip_hi <= x)
        # assert torch.all((where_x_lo + where_x_nc + where_x_hi) == 1.0)

        # rescale by the quantum to prepare for integerisation
        # `eps / 4` is arbitrary: any value between zero and `eps / 2` can
        # guarantee proper saturation both with the flooring and the rounding
        # operations.
        x_scaled_and_clipped = (x.clamp(clip_lo, clip_hi + (eps / 4)) - clip_lo) / eps

        # integerise
        x_int = x_scaled_and_clipped.floor() if floor else x_scaled_and_clipped.round()

        # fake-quantise
        x_fq = clip_lo + eps * x_int

        # compute the quantisation error
        qerr = x_fq - x

        # pack context
        ctx.save_for_backward(where_x_lo, where_x_nc, where_x_hi, clip_lo, clip_hi,
                              qerr, beta, beta_running, g_log_t_running_var, clip_g_log_t,
                              clip_g)

        return x_fq

    @staticmethod
    def backward(ctx, g_in: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None, torch.Tensor, None, None, None, None, None]:

        # unpack context
        where_x_lo, where_x_nc, where_x_hi, clip_lo, clip_hi, qerr, beta, beta_running, g_log_t_running_var, clip_g_log_t, clip_g = ctx.saved_variables

        # I define these constants once to avoid recreating and casting `torch.Tensor`s at each place where they're needed
        zero = torch.zeros(1).to(where_x_nc.device)
        ln2 = torch.log(torch.Tensor([2.0]).to(where_x_nc.device))
        g_eps = torch.Tensor([1e-5]).to(where_x_nc.device)

        # clip the gradient that goes towards the input?
        # See "Quantized neural networks: training neural networks with low
        # precision weights and activations" (2018), Hubara et al.,
        # Section 2.3, equation #6.
        g_out = torch.where(where_x_nc, g_in, zero) if clip_g else g_in

        # allow channel-wise learnable quanta
        reduce_dims = tuple(range(g_in.ndim))
        reduce_dims = reduce_dims[1:] if clip_lo.ndim > 1 else reduce_dims

        # compute the gradient that goes towards the thresholds
        g_log_t = clip_lo * torch.where(where_x_lo, g_in, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        g_log_t += torch.where(where_x_nc, qerr * g_in, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        g_log_t += clip_hi * torch.where(where_x_hi, g_in, zero).sum(dim=reduce_dims).reshape(clip_hi.shape)
        g_log_t *= ln2

        # normalize the gradient that goes towards the thresholds
        # See "Trained quantization thresholds for accurate and efficient
        # fixed-point inference of deep neural networks" (2019), Jain et al.,
        # Appendix B.2, equation #17.
        g_log_t_running_var_temp = beta * g_log_t_running_var + (1 - beta) * (g_log_t ** 2)
        g_log_t_running_var.copy_(g_log_t_running_var_temp.reshape(g_log_t_running_var.shape))
        beta_running.mul_(beta)
        g_log_t = g_log_t / (torch.sqrt(g_log_t_running_var_temp / (1 - beta_running)) + g_eps)

        # clip the gradient that goes towards the thresholds?
        # See "Trained quantization thresholds for accurate and efficient
        # fixed-point inference of deep neural networks" (2019), Jain et al.,
        # Appendix B.2, equation #18.
        g_log_t = torch.tanh(g_log_t) if clip_g_log_t else g_log_t

        #      x      eps   clip_lo clip_hi floor, log_t    beta  beta_running g_log_t_running_var clip_g_log_t clip_g
        return g_out, None, None,   None,   None,  g_log_t, None, None,        None,               None,        None
