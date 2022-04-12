from typing import Optional

import numpy as np

import torch

from quantlib.algorithms.pact import PACTQuantize
from quantlib.algorithms.pact.pact_functions import AlmostSymmQuantFunc

# Bayesian Bits quantization as described in "Bayesian Bits: Unifying
# Quantization and Pruning", see https://arxiv.org/abs/2005.07093


def bb_get_gates(phi : torch.Tensor, lo : float, hi : float, T : float, expand : Optional[int] = None):
    # if we want to do dimension-wise expansion (e.g., each output channel gets
    # its own set of gates), u must be expanded and phi must be unsqueezed
    if expand is not None:
        u_size = phi.size() + (expand,)
        phi = phi.unsqueeze(1)
    else:
        u_size = phi.size()
    u = torch.rand(size=u_size, device=phi.device)
    g = torch.log(u/(1-u))
    s = torch.sigmoid((g+phi)/T)
    s_scaled = s * (hi-lo) + lo
    z = torch.min(torch.ones([], device=phi.device), torch.max(torch.zeros([], device=phi.device), s_scaled))
    return z

#cumulative distribution function (eq. 21 in paper, eqs. 24, 25
#in the "hard concrete" paper)
def bb_cdf(s : float, phi : torch.Tensor, lo : float, hi : float, T : float):
    inner = torch.log((s - lo)/(hi - s)) * T - phi
    return torch.sigmoid(inner)

# complementary CDF: 1-CDF
def bb_ccdf(s : float, phi : torch.Tensor, lo : float, hi : float, T : float):
    # sigm(x) = 1 - sigm(-x)
    inner = phi - torch.log((s - lo)/(hi - s)) * T
    return torch.sigmoid(inner)

def BBQuantize(inp : torch.Tensor, phi : torch.Tensor, hc_lo : float, hc_hi : float, T : float, clip_lo : torch.Tensor, clip_hi : torch.Tensor, precs : list, symm : bool, expand : bool = False):
    n_phi = phi.numel()
    assert n_phi == len(precs) - 1, "Phi should contain one fewer parameter than # of precisions!"
    precs = sorted(precs)
    levels = [int(2**p) for p in precs]
    expand_factor = inp.shape[0] if expand else None
    if symm:
        # detach!
        clip_hi = [AlmostSymmQuantFunc.apply(clip_lo, l) for l in levels]
    else:
        clip_hi = [clip_hi] * len(levels)
    if len(clip_hi) > 1:
        clip_hi = [clip_hi[0]] + [c_h.detach() for c_h in clip_hi[1:]]

    eps = [((h-clip_lo)/(lvl-1)).detach() for h, lvl in zip(clip_hi, levels)]


    clip_lo = [clip_lo] + [clip_lo.detach()] * (len(levels)-1)
    inp_q = torch.stack([PACTQuantize(inp, e, c_l, c_h, floor=False, clip_gradient=torch.tensor(True, device=inp.device), noisy=False) for e, c_l, c_h in zip(eps, clip_lo, clip_hi)])
    errs = inp_q[1:] - inp_q[:-1]
    errs = errs.detach()

    gates = bb_get_gates(phi, hc_lo, hc_hi, T, expand=expand_factor)
    if expand:
        gates = gates.reshape(gates.size()+(1,)*(inp.ndim-1))

    # no pruning for now
    out_q = inp_q[0]
    chained_gates = torch.cumprod(gates, 0)
    #print(f"chained_gates: {chained_gates}")
    for i in range(len(gates)):
        out_q = out_q + errs[i]*chained_gates[i]
    #out_q = out_q + torch.sum(errs * chained_gates)

    return out_q

def BBQuantizeTestTime(inp : torch.Tensor, phi : torch.Tensor, hc_lo : float, hc_hi : float, T : float, clip_lo : torch.Tensor, clip_hi : torch.Tensor, precs : list, symm : bool):

    gates = torch.cat((torch.tensor([True], device=inp.device),  bb_cdf(torch.zeros([], device=inp.device), phi, hc_lo, hc_hi, T) < 0.34))

    levels = [int(2**p) for p in precs]
    if symm:
        clip_hi = [AlmostSymmQuantFunc.apply(clip_lo, l) for l in levels]
    else:
        clip_hi = [clip_hi] * len(levels)

    for i, g in enumerate(gates):
        if g:
            n_levels = levels[i]
            clip_upper = clip_hi[i]
        else:
            break
    eps = (clip_upper - clip_lo)/(n_levels-1)

    return PACTQuantize(inp, eps, clip_lo, clip_upper, floor=False, clip_gradient=torch.tensor(True, device=inp.device), noisy=False)
