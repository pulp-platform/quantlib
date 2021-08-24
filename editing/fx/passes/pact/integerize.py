from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
from functools import partial

import numpy as np

import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *

from .. import FxPass, ReplaceSequentialPatternPass, ModifySequentialPatternPass, SequentialPass, ShapePropPass
from .. import AnnotateEpsPass, extract_eps
from .. import MergeConvBNPass, RetracePass
from ...util import gm_modules, module_of_node
from ...util.tracing import LeafTracer, custom_symbolic_trace

from .pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace

__all__ = ['IntegerizePACTConvPass',
           'IntegerizePACTLinearPass',
           'IntegerizeBNActPass',
           'IntegerizePACTNetPass',
           'PACTTracer',
           'PACT_symbolic_trace',
           'RequantShift']

class RequantShift(nn.Module):
    def __init__(self, mul : torch.Tensor, add : torch.Tensor, n_levels : int, signed : bool = False, D : torch.Tensor = torch.tensor(2**24)):
        super(RequantShift, self).__init__()
        self.register_buffer('mul', mul)
        self.register_buffer('add', add)
        self.register_buffer('div', D)
        self.signed = signed
        self.n_levels_out = n_levels

    def forward(self, x):
        x = x * self.mul
        x = x + self.add
        x = (x/self.div).floor()
        if not self.signed:
            x = torch.clip(x, 0., float(self.n_levels_out-1))
        else:
            c = np.floor(self.n_levels_out/2+0.001)
            if self.n_levels_out % 2:
                x = torch.clip(x, -c, c)
            else:
                x = torch.clip(x, -c, c-1)
        return x

def integerize_pact_conv_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    conv = matched_modules[0]
    assert isinstance(conv, (PACTConv1d, PACTConv2d)), f"integerize_pact_conv_fun got bad match - expected PACTConv, got {type(conv)}"
    assert conv.bias is None, "integerize_pact_conv_fun: Conv layer has bias"

    conv_type = nn.Conv2d if isinstance(conv, PACTConv2d) else nn.Conv1d
    new_conv = conv_type(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=False,
                         padding_mode=conv.padding_mode)
    new_conv.weight.data.copy_(conv.weight_int)

    # annotate the new conv with the number of levels
    new_conv.n_levels = conv.n_levels

    return new_conv


class IntegerizePACTConvPass(Sequentialpass):
    def __init__(self):
        passes = []
        for i, c in enumerate((PACTConv1d, PACTConv2d)):
            pattern = nn.Sequential(c(1,1,1))
            name = f"_INTEGERIZE_PACT_CONV{i+1}D_PASS"
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, integerize_pact_conv_fun, name))
        super(IntegerizePACTConvPass, self).__init__(*passes, name_prefix='_INTEGERIZE_PACT_CONVS_PASS')

def integerize_pact_linear_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    lin_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    lin = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    assert isinstance(lin, PACTLinear), f"integerize_pact_linear_fun got bad match - expected PACTLinear, got {type(lin)}"
    # note the new node's intended integer precision in the precision dict
    #if prec_dict is not None:
        #nbits = int(np.log2(lin.n_levels) + 0.2)
#        prec_dict[name] = nbits

    new_lin = nn.Linear(in_features=lin.in_features,
                        out_features=lin.out_features,
                        bias=(lin.bias is not None))
    new_lin.weight.data.copy_(lin.weight_int.round())
    if lin.bias is not None:
        new_lin.bias.data.copy_(lin.get_bias_int(eps_in).round())

    new_lin.n_levels = lin.n_levels

    return new_lin

class IntegerizePACTLinearPass(ReplaceSequentialPatternPass):
    def __init__(self):
        pattern = nn.Sequential(PACTLinear(1,1))
        name = "_INTEGERIZE_PACT_LIN_PASS"
        super(IntegerizePACTLinearPass, self).__init__(pattern, PACT_symbolic_trace, integerize_pact_linear_fun, name)


def bn_act_to_requant_fun(gm : fx.GraphModule, match : Match, D=2**24):
    modules = dict(gm.named_modules())
    if not isinstance(D, torch.Tensor):
        D = torch.tensor(D)
    matched_nodes = [n for n in match.nodes_map.values()][-2:0:-1]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    assert len(matched_nodes) == len(matched_modules), "bn_act_to_requant got unexpected non-'call_module' nodes!"
    if len(matched_modules) == 1:
        act = matched_modules[0]
        act_node = matched_nodes[0]
        bn = None
    else:
        assert len(matched_modules) == 2, "bn_act_to_requant expected match of length 1 or 2!"
        act = matched_modules[1]
        act_node = matched_nodes[1]
        bn = matched_modules[0]
        bn_node = matched_nodes[0]
        assert isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), f"bn_act_to_requant called on incompatible BN layer {type(bn)}"
    assert isinstance(act, (PACTUnsignedAct, PACTAsymmetricAct)), f"bn_act_to_requant called on incompatible activation {type(act)}"

    signed_act = isinstance(act, PACTAsymmetricAct)
    eps_in = extract_eps(act_node.meta['quant'].eps_in).cpu().clone().detach().squeeze()
    eps_out = act_node.meta['quant'].eps_out.cpu().clone().detach().squeeze()

    # if the requant node would perform an identity operation, don't insert it.
    if eps_in.numel() == eps_out.numel() == 1 and eps_in == eps_out and bn is None:
        return None

    gamma_h = (bn.weight/torch.sqrt(bn.running_var+bn.eps)) if bn is not None else torch.ones_like(eps_in)
    beta_h = bn.bias - bn.running_mean * gamma_h if bn is not None else torch.zeros_like(gamma_h)
    gamma_h *= eps_in
    gamma_h /= eps_out
    beta_h /= eps_out
    if act.rounding:
        beta_h += 0.5

    gamma_h *= D
    beta_h *= D
    gamma_h = torch.round(gamma_h)
    beta_h = torch.round(beta_h)

    if bn and gamma_h.numel() > 1:
        if isinstance(bn, nn.BatchNorm1d):
            # BN1D can take two input tensor formats:
            # 1. [N_B, N_C, L], e.g. if it comes after a Conv1d layer
            # 2. [N_B, L], e.g. if it comes after a linear layer
            # depending on how it is used in the network we are processing, the
            # mul/add parameters must have different shape. we use the
            # information stored in the graph by the ShapePropPass to determine
            # this.
            bn_outshape = bn_node.meta['tensor_meta'].shape
            if len(bn_outshape) == 3:
                gamma_h = gamma_h.reshape((gamma_h.numel(), 1))
                beta_h = beta_h.reshape((beta_h.numel(), 1))
            elif len(bn_outshape) == 2:
                gamma_h = gamma_h.reshape((gamma_h.numel(),))
                beta_h = beta_h.reshape((beta_h.numel(),))
        elif isinstance(bn, nn.BatchNorm2d):
            gamma_h = gamma_h.reshape((gamma_h.numel(), 1, 1))
            beta_h = beta_h.reshape((beta_h.numel(), 1, 1))
        elif isinstance(bn, nn.BatchNorm3d):
            gamma_h = gamma_h.reshape((gamma_h.numel(), 1, 1, 1))
            beta_h = beta_h.reshape((beta_h.numel(), 1, 1, 1))

    requant = RequantShift(gamma_h, beta_h, act.n_levels, signed_act, D)
    return requant

class IntegerizeBNActPass(SequentialPass):
    def __init__(self, D : float = 2**24):
        passes = []
        # replace all combinations of BN + PACT activation with RequantShift layers
        for act_name, act_type in [("UNSIGNED_ACT", PACTUnsignedAct), ("SIGNED_ACT", PACTAsymmetricAct)]:
            for bn_name, bn_type in [("BN1D", nn.BatchNorm1d), ("BN2D", nn.BatchNorm2d), ("BN3D", nn.BatchNorm3d)]:
                pattern = nn.Sequential(bn_type(1), act_type())
                passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, bn_act_to_requant_fun, f"_INTEGERIZE_{bn_name}_{act_name}_PASS", D=D))
            #also replace "freestanding" activations AFTER replacing the BN+Act stacks
            pattern = nn.Sequential(act_type())
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, bn_act_to_requant_fun, f"_INTEGERIZE_{act_name}_PASS", D=D))

        super(IntegerizeBNActPass, self).__init__(*passes, name_prefix="_INTEGERIZE_BN_ACT_PASS")

class IntegerizePACTNetPass(SequentialPass):
    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size], eps_in : Optional[Union[torch.Tensor, float]] = None, D : float = 2**24):
        passes = []
        # start by retracing the network to dissolve any integer ops
        passes.append(RetracePass(PACT_symbolic_trace))
        # then run a shape propagation pass so the conversion functions can
        # know what shape a node's output has
        #IMPORTANT: run model.eval() BEFORE running this pass - otherwise the
        # ShapePropPass will contaminate the batchnorm parameters!
        passes.append(ShapePropPass(shape_in))
        # first step: merge any convolutions with biases into batch norms
        passes.append(MergeConvBNPass(PACT_symbolic_trace))
        # second step: annotate epsilons
        passes.append(AnnotateEpsPass(eps_in))
        # with epsilons annotated everywhere, we can integerize linear
        # functions (conv and FC)
        passes.append(IntegerizePACTConvPass())
        passes.append(IntegerizePACTLinearPass())
        # now, PACT Activations (combined with BN layers) can be converted to
        # RequantShift layers
        passes.append(IntegerizeBNActPass(D))
        super(IntegerizePACTNetPass, self).__init__(*passes, name_prefix="_INTEGERIZE_PACT_NET_PASS")
