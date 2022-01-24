from typing import Optional

import torch
from torch import nn

from quantlib.algorithms.bb import BBAct, BBConv2d, BBLinear


def madd_of_conv(insize, c):
    madds_per_pixel = c.kernel_size[0]*c.kernel_size[1]*c.in_channels*c.out_channels//c.groups
    return madds_per_pixel * insize * insize//(c.stride[0]*c.stride[1])
def madd_of_lin(l):
    return l.ixn_features * l.out_features

class BBGateController:
    def __init__(self, layer : nn.Module, loss_scale : Optional[float] = None):
        # 0. Sanity check
        assert isinstance(layer, (BBConv2d, BBLinear, BBAct)), "Layer of BBActConvGateController must be BBAct, BBLinear, BBConv1d or BBConv2d!"

        # 1. set up fields
        self.layer = layer
        self.loss_scale = 1. if loss_scale is None else loss_scale

        # 2. set up gating parameters of conv and act layer
        n_precs = len(layer.precs)

        assert "bb_gates" not in dict(layer.named_parameters()), "Expected linear layer to not have 'bb_gates' parameter!"
        gates = nn.Parameter(torch.full((n_precs-1,), 2.))
        layer.bb_gates = gates
        layer.register_gate_ctrl(self)
        self.gates = gates

    def loss_term(self):
        ccdfs = self.layer.ccdf0()
        # ehehe cumprod :^)
        cumulative_ccdfs = torch.cumprod(ccdfs, 0)
        gated_precs = torch.tensor(self.layer.precs[1:], device=cumulative_ccdfs.device)
        loss_term = torch.sum(cumulative_ccdfs * gated_precs)

        return self.loss_scale * loss_term
