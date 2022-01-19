import torch
from typing import Tuple


class PACTQuantiser(torch.autograd.Function):
    """PACT (PArametrized Clipping acTivation) quantization function.

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
                eps: torch.Tensor,
                clip_lo: torch.Tensor,
                clip_hi: torch.Tensor,
                noisy: bool = False,
                floor: bool = True,
                clip_g: bool = True) -> torch.Tensor:
        """Compute the forward pass of the PACT op.

        Arguments:
            x: the array to be quantized.
            eps: the (precomputed) value of :math:`\varepsilon`.
            clip_lo: the lower clipping bound :math:`\alpha`.
            clip_hi: the upper clipping bound :math:`beta`.
            noisy: whether or not to add some random (uniformly distributed) noise
                to the scaled-and-clipped version of the array (before the
                integerisation op).
            floor: whether to apply flooring or not (therefore using rounding).
            clip_g: whether zeroing the components of the incoming gradient array
                if the corresponding components of the input array are outside the
                clipping range :math:`[\alpha, \beta)`.

        Returns:
            x_int: the fake-quantised array.

        """

        # partition the components of the input tensor with respect to the clipping bounds
        where_x_lo = (x < clip_lo)
        where_x_nc = (clip_lo <= x) * (x < clip_hi)  # non-clipped
        where_x_hi = (clip_hi <= x)
        # assert torch.all((where_x_lo + where_x_nc + where_x_hi) == 1.0)

        # for completeness' sake (e.g. to reproduce the results from the
        # PACT+SAWB paper), we allow for outputs which are not an integer multiple of
        # eps.
        # to ensure hardware compatibility, it is the downstream user's
        # responsibility to ensure that clip_lo/clip_hi are multiples of eps!

        # x_scaled_and_clipped = (x.clamp(clip_lo, clip_hi) - clip_lo) / eps  # vanilla
        # x_scaled_and_clipped = (input.clamp(clip_lo, clip_hi + 1e-7) - clip_lo) / eps  # workaround to FP instability (alternative 1)
        x_scaled_and_clipped = (x / eps).clamp(clip_lo / eps, clip_hi / eps) - clip_lo / eps  # workaround to FP instability (alternative 2)

        # maybe add noise
        if noisy:
            x_scaled_and_clipped += torch.rand_like(x_scaled_and_clipped) - 0.5

        # integerised version (fused binning and re-mapping)
        x_int = x_scaled_and_clipped.floor() if floor else x_scaled_and_clipped.round()

        # fake-quantised
        x_fq = clip_lo + x_int * eps

        # pack context
        ctx.save_for_backward(where_x_lo, where_x_nc, where_x_hi, clip_lo, clip_hi, clip_g)

        return x_fq

    @staticmethod
    def backward(ctx, g_in: torch.Tensor) -> Tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None, None, None]:

        # unpack context
        where_x_lo, where_x_nc, where_x_hi, clip_lo, clip_hi, clip_g = ctx.saved_variables

        zero = torch.zeros(1).to(where_x_nc.device)

        # clip the gradient?
        # See "Quantized neural networks: training neural networks with low
        # precision weights and activations", Hubara et al., Section 2.3,
        # equation #6.
        g_out = torch.where(where_x_nc, g_in, zero) if clip_g else g_in

        # allow channel-wise learnable quanta
        reduce_dims = tuple(range(g_in.ndim))
        reduce_dims = reduce_dims[1:] if clip_lo.ndim > 1 else reduce_dims

        # gradients to the clipping bounds
        g_clip_lo = torch.where(where_x_lo, g_in, zero).sum(dim=reduce_dims).reshape(clip_lo.shape)
        g_clip_hi = torch.where(where_x_hi, g_in, zero).sum(dim=reduce_dims).reshape(clip_hi.shape)

        return g_out, None, g_clip_lo, g_clip_hi, None, None, None
