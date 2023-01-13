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


class _PACTQuantiser(torch.autograd.Function):
    """PACT (PArametrized Clipping acTivation) quantisation function.

    This function acts component-wise on the input array. In the forward pass,
    it applies the following operation:

    .. math::
       f(x) \coloneqq \varepsilon \left\lfloor \clip_{[\alpha, \beta)} \left( \frac{x}{\varepsilon} \right) \right\rfloor \,,

    where :math:`\clip_{ [\alpha, \beta) }` is the clipping function

    .. math::
       \clip_{ [\alpha, \beta) }(t) \coloneqq \max \{ \alpha, \min\{ t, \beta \} \}` \,,

    and :math:`\varepsilon \coloneqq (\beta - \alpha) / (K - 1)` is the
    *precision* or *quantum* of the quantiser (:math:`K > 1` is an integer).
    It is possible to replace flooring with rounding:

    .. math::
       f(x) \coloneqq \varepsilon \left\lfloor \clip_{[\alpha, \beta)} \left( \frac{x}{\varepsilon} \right) \right\rceil \,.

    In the backward pass, it applies the straight-through estimator (STE)

    .. math::
       \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \,,

    where :math:`y \coloneqq f(x)`. Possibly the gradient is clipped to the
    clipping range :math:`[\alpha, \beta)` applied during the forward pass:

    .. math::
       \frac{partial L}{\partial x} = \frac{\partial L}{\partial y} \chi_{[\alpha, \beta)}(x) \,.

    """

    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                clip_lo: torch.Tensor,
                clip_hi: torch.Tensor,
                step: torch.Tensor,
                scale: torch.Tensor,
                round: bool = False,
                clip_g: bool = True) -> torch.Tensor:
        """Compute the forward pass of the PACT operation.

        Arguments:
            x: the array to be quantised.
            clip_lo: the lower clipping bound :math:`\alpha`.
            clip_hi: the upper clipping bound :math:`beta`.
            step: the number of "unit steps" to take between a quantisation
                threshold and the successive one.
            scale: the (precomputed) value of :math:`\varepsilon`.
            round: whether to apply rounding (True) or flooring (False) to
                integerise the array.
            clip_g: whether zeroing the components of the outgoing gradient
                array if the corresponding components of the input array are
                outside the clipping range :math:`[\alpha, \beta)`.

        Returns:
            x_fq: the fake-quantised array.

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
        x_scaled_and_clipped = torch.clamp(x - clip_lo, torch.tensor(0.0).to(device=clip_lo.device), clip_hi + (scale / 4) - clip_lo) / (step * scale)  # TODO: is `clip_lo` a reasonably reliable reference to move a newly-generated tensor to a specific device?

        # integerise (fused binning and re-mapping)
        x_int = (x_scaled_and_clipped + 0.5).floor() if round else x_scaled_and_clipped.floor()

        # fake-quantise
        x_fq = clip_lo + scale * x_int

        # pack context
        ctx.save_for_backward(where_x_lo, where_x_nc, where_x_hi, clip_lo, clip_hi, torch.tensor(clip_g))

        return x_fq

    @staticmethod
    def backward(ctx, g_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        """Compute the backward pass of the PACT operation."""

        # unpack context
        where_x_lo, where_x_nc, where_x_hi, clip_lo, clip_hi, clip_g = ctx.saved_tensors

        # I define this constant once to avoid recreating and casting a `torch.Tensor` at each place where it's needed
        zero = torch.zeros(1).to(where_x_nc.device)

        # clip the gradient that goes towards the input?
        # See "Quantized neural networks: training neural networks with low
        # precision weights and activations", Hubara et al., Section 2.3,
        # equation #6.
        g_out = torch.where(where_x_nc, g_in, zero) if clip_g else g_in

        # gradients to the clipping bounds
        reduce_dims = tuple(i for i, d in enumerate(clip_lo.shape) if d == 1) if clip_lo.shape != (1,) else tuple(range(0, g_in.ndim))  # respect granularity
        g_clip_lo = torch.where(where_x_lo, g_in, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        g_clip_hi = torch.where(where_x_hi, g_in, zero).sum(dim=reduce_dims).reshape(clip_hi.shape)

        #      x      clip_lo    clip_hi    step  scale floor clip_g
        return g_out, g_clip_lo, g_clip_hi, None, None, None, None
