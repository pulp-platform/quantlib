import torch
from torch import nn
from quantlib.algorithms.bb.bb_functions import BBQuantize, BBQuantizeTestTime, bb_ccdf, bb_cdf
from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_ops import _PACTActivation
__all__ = ["_BB_CLASSES",
           "_BB_LINOPS",
           "BBAct",
           "BBConv2d",
           "BBLinear"]





class BBConv2d(PACTConv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 precs,
                 hc_stretch,
                 hc_T,
                 quantize,
                 init_clip, #TODO: make this 'max' always?
                 **kwargs):

        super(BBConv2d, self).__init__(in_channels,
                                       out_channels,
                                       kernel_size,
                                       256, #n_levels is not used
                                       quantize=quantize,
                                       init_clip=init_clip,
                                       learn_clip=False,
                                       **kwargs)
        assert hc_stretch >= 1., "BBConv2d: Hard concrete stretch factor must be >= 1!"
        self.hc_stretch = hc_stretch
        hc_over = (hc_stretch - 1.)/2
        self.precs = precs
        self.hc_lo = -hc_over
        self.hc_hi = 1. + hc_over
        self.hc_T = hc_T
        self.gate_ctrl = None
        self.bb_gates = None

    def register_gate_ctrl(self, c):
        if self.gate_ctrl is not None:
            print("Warning: BBConv2d's gate_ctrl is being overwritten... This is probably due to a bug in your code!")
        self.gate_ctrl = c


    @property
    def weight_q(self):
        if self.training:
            #return PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, self.clip_hi, floor=False, clip_gradient=self.clip_gradient)
            # stochastic quantized weights
            return BBQuantize(self.weight, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.symm_wts, expand=True)
        else:
            # deterministically quantized weights
            return BBQuantizeTestTime(self.weight, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.symm_wts)

    def ccdf0(self):
        return bb_ccdf(torch.zeros([]), self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T)

    def extra_repr(self):
        r = nn.Conv2d.extra_repr(self)
        r += f", precs={self.precs}, hc_stretch={self.hc_stretch}, hc_T={self.hc_T}"
        r += self.pact_repr_str
        return r

    def get_n_levels(self):
        gates = torch.cat((torch.tensor([True], device=self.bb_gates.device),  bb_cdf(torch.zeros([], device=self.bb_gates.device), self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T) < 0.34))

        for i, g in enumerate(gates):
            if g:
                n_levels = int(2**self.precs[i])
            else:
                break
        return n_levels


class BBLinear(PACTLinear):
    def __init__(self,
                 in_features,
                 out_features,
                 precs,
                 hc_stretch,
                 hc_T,
                 quantize,
                 init_clip,
                 **kwargs):

        super(BBLinear, self).__init__(in_features,
                                         out_features,
                                         256,
                                         quantize=quantize,
                                         init_clip=init_clip,
                                         learn_clip=False,
                                         **kwargs)
        assert hc_stretch >= 1., "BBLinear: Hard concrete stretch factor must be >= 1!"
        self.hc_stretch = hc_stretch
        hc_over = (hc_stretch - 1.)/2
        self.precs = precs
        self.hc_lo = -hc_over
        self.hc_hi = 1. + hc_over
        self.hc_T = hc_T
        self.gate_ctrl = None
        self.bb_gates = None

    def register_gate_ctrl(self, c):
        if self.gate_ctrl is not None:
            print("Warning: BBLinear's gate_ctrl is being overwritten... This is probably due to a bug in your code!")
        self.gate_ctrl = c


    @property
    def weight_q(self):
        if self.training:
            #return PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, self.clip_hi, floor=False, clip_gradient=self.clip_gradient)
            # stochastic quantized weights
            return BBQuantize(self.weight, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.symm_wts, expand=True)
        else:
            # deterministically quantized weights
            return BBQuantizeTestTime(self.weight, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.symm_wts)


    def ccdf0(self):
        return bb_ccdf(torch.zeros([]), self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T)

    def extra_repr(self):
        r = nn.Linear.extra_repr(self)
        r += f", precs={self.precs}, hc_stretch={self.hc_stretch}, hc_T={self.hc_T}"
        r += self.pact_repr_str
        return r

    def get_n_levels(self):
        gates = torch.cat((torch.tensor([True], device=self.bb_gates.device),  bb_cdf(torch.zeros([], device=self.bb_gates.device), self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T) < 0.34))

        for i, g in enumerate(gates):
            if g:
                n_levels = int(2**self.precs[i])
            else:
                break
        return n_levels

class BBAct(_PACTActivation):
    def __init__(self,
                 precs,
                 hc_stretch,
                 hc_T,
                 init_clip,
                 learn_clip,
                 act_kind,
                 signed,
                 leaky=0.1):
        super(BBAct, self).__init__(256,
                                    init_clip,
                                    learn_clip,
                                    act_kind,
                                    symm=signed,
                                    rounding=True,
                                    signed=signed)
        assert hc_stretch >= 1., "BBAct: Hard concrete stretch factor must be >= 1!"
        self.hc_stretch = hc_stretch
        hc_over = (hc_stretch - 1.)/2
        self.precs = precs
        self.hc_lo = -hc_over
        self.hc_hi = 1. + hc_over
        self.hc_T = hc_T
        self.gate_ctrl = None
        self.bb_gates = None

    def register_gate_ctrl(self, c):
        if self.gate_ctrl is not None:
            print("Warning: BBAct's gate_ctrl is being overwritten... This is probably due to a bug in your code!")
        self.gate_ctrl = c

    def forward(self, x):
        if not self.started:
            return super(BBAct, self).forward(x)
        elif self.training:
            # expand in batch dimension (!!!) Does it make sense? who knows...
            xb = BBQuantize(x, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.signed, expand=True)
            #print(f"diff to pact: {(xb-xp).abs().mean()}")

            return xb
            #return xp
        else:
            return BBQuantizeTestTime(x, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.signed)

    def ccdf0(self):
        return bb_ccdf(torch.zeros([]), self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T)

    def extra_repr(self):
        r = super(BBAct, self).extra_repr()
        r += f", precs={self.precs}, hc_stretch={self.hc_stretch}, hc_T={self.hc_T}"
        return r

    def get_n_levels(self):
        gates = torch.cat((torch.tensor([True], device=self.bb_gates.device),  bb_cdf(torch.zeros([], device=self.bb_gates.device), self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T) < 0.34))

        for i, g in enumerate(gates):
            if g:
                n_levels = int(2**self.precs[i])
            else:
                break
        return n_levels

_BB_CLASSES = [BBAct,
               BBConv2d,
               BBLinear]

_BB_LINOPS = [BBConv2d,
              BBLinear]
