from typing import Union
import torch
from torch import nn
from quantlib.algorithms.bb.bb_functions import BBQuantize, BBQuantizeTestTime, bb_ccdf, bb_cdf
from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_ops import _PACTActivation
__all__ = ["_BB_CLASSES",
           "_BB_LINOPS",
           "BBAct",
           "BBConv2d",
           "BBLinear",
           "BBIntegerAdd"]




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

        if "n_levels" in kwargs.keys():
            del kwargs["n_levels"]
        if "learn_clip" in kwargs.keys():
            del kwargs["learn_clip"]
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
        self.gate_ctrls = []
        self.bb_gates = None


    def register_gate_ctrl(self, c):
#        if self.gate_ctrl is not None:
 #           print("Warning: BBConv2d's gate_ctrl is being overwritten... This is probably due to a bug in your code!")
        self.gate_ctrls.append(c)


    @property
    def weight_q(self):
        if self.training:
            #return PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, self.clip_hi, floor=False, clip_gradient=self.clip_gradient)
            # stochastic quantized weights
            return BBQuantize(self.weight, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.symm_wts, expand=False)
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

        if "n_levels" in kwargs.keys():
            del kwargs["n_levels"]
        if "learn_clip" in kwargs.keys():
            del kwargs["learn_clip"]
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
        self.gate_ctrls = []
        self.bb_gates = None

    def register_gate_ctrl(self, c):
        #if self.gate_ctrl is not None:
            #print("Warning: BBLinear's gate_ctrl is being overwritten... This is probably due to a bug in your code!")
        self.gate_ctrls.append(c)


    @property
    def weight_q(self):
        if self.training:
            #return PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, self.clip_hi, floor=False, clip_gradient=self.clip_gradient)
            # stochastic quantized weights
            return BBQuantize(self.weight, self.bb_gates, self.hc_lo, self.hc_hi, self.hc_T, self.clip_lo, self.clip_hi, self.precs, self.symm_wts, expand=False)
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
                 leaky=0.1,
                 nb_std=3):

        super(BBAct, self).__init__(256,
                                    init_clip,
                                    learn_clip,
                                    act_kind,
                                    symm=signed,
                                    rounding=True,
                                    signed=signed,
                                    nb_std=nb_std)
        assert hc_stretch >= 1., "BBAct: Hard concrete stretch factor must be >= 1!"
        self.hc_stretch = hc_stretch
        hc_over = (hc_stretch - 1.)/2
        self.precs = precs
        self.hc_lo = -hc_over
        self.hc_hi = 1. + hc_over
        self.hc_T = hc_T
        self.gate_ctrls = []
        self.bb_gates = None

    def register_gate_ctrl(self, c):
        #if self.gate_ctrl is not None:
         #   print("Warning: BBAct's gate_ctrl is being overwritten... This may be due to a bug in your code, or you have multiple controllers for the same layer (which may be legitimate)")
        self.gate_ctrls.append(c)

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


class BBIntegerAdd(PACTIntegerAdd):
    def __init__(self,
                 num_args = 1,
                 signed : Union[bool, list] = True,
                 pact_kwargs : dict = {},
                 bb_kwargs : dict = {}):

        nn.Module.__init__(self)
        if isinstance(signed, bool):
            signed = [signed] * (num_args + 1)

        assert len(signed) == num_args + 1, f"BBIntegerAdd expected {num_args+1} elements in 'signed', got {len(signed)}"
        self.acts = torch.nn.ModuleList([])
        for i in range(num_args):
            in_act_cls = PACTAsymmetricAct if signed[i] else PACTUnsignedAct
            self.acts.append(in_act_cls(**pact_kwargs))

        self.clip_lo = self.acts[0].clip_lo
        self.clip_hi = self.acts[0].clip_hi
        self.n_levels = self.acts[0].n_levels
        self.force_out_eps = False
        self.act_out = BBAct(signed=signed[-1], **bb_kwargs)
