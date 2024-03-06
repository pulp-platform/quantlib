#
# integerize.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Victor Jung <jungvi@iis.ee.ethz.ch>
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

from typing import Union, Optional, Literal
from functools import partial
from copy import deepcopy
import numpy as np
import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_ops import RequantShift, HardActRequantShift, ChannelwiseThreshold
from .harmonize import InsertBNBetweenBiasedConvAndActsPass, RQSMergePass
from .pact_util import PACTTracer, PACT_symbolic_trace
from ...util import gm_modules, module_of_node, get_ordered_active_nodes
from .. import FxPass, ReplaceSequentialPatternPass, SequentialPass, ShapePropPass, ModularizePass
from .. import AnnotateEpsPass, extract_eps
from .. import MergeConvBNPass, RetracePass
from .harmonize import LayerNormDisassemblePass, ApplyPassToWrapModule, InsertBNBetweenBiasedConvAndActsPass, RQSMergePass
from ...util import gm_modules, module_of_node, get_ordered_active_nodes, modules_of_match
from ...util.tracing import LeafTracer, custom_symbolic_trace

from quantlib.algorithms.pact.pact_ops import RequantShift, HardActRequantShift, ChannelwiseThreshold


__all__ = ['IntegerizePACTConvPass',
           'IntegerizePACTLinearPass',
           'IntegerizeBNActPass',
           'IntegerizePACTNetPass',
           'IntegerizeSoftmaxPass',
           'IntegerizeGELUPass',
           'IntegerizeLayerNormPass',
           'IntegerizeEmbeddingsPass',
           'FixChannelNumbersPass',
           'IntegerizeBNPACTHardActsPass',
           'PACTTracer',
           'PACT_symbolic_trace',]

def integerize_softmax_fun(gm : fx.GraphModule, match : Match, mode: Literal["I-BERT", "ITA", 'ITA-Partial'] = "I-BERT", D=2**12, export_node=False):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    lin_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    # assert isinstance(module, PACTSoftmax), f"integerize_softmax_fun got bad match - expected PACTSoftmax, got {type(module)}"

    if mode=='I-BERT':
        new_softmax = PACTIntegerSoftmax(n_levels=module.n_levels, eps_in=eps_in, export_node=export_node)
    elif mode=='ITA':
        new_softmax = PACTIntegerITAMax(max_value = module.act.max, n_levels=module.n_levels, eps_in=eps_in, D=D, export_node=export_node)
    elif mode=='ITA-Partial':
        new_softmax = PACTIntegerITAPartialMax(max_value = module.act.max, n_levels=module.n_levels, eps_in=eps_in, D=D, export_node=export_node)
    else:
        assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"

    return new_softmax

def integerize_gelu_fun(gm : fx.GraphModule, match : Match, D=2**14, export_node = False):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    lin_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    assert isinstance(module, PACTGELU), f"integerize_gelu_fun got bad match - expected PACTGELU, got {type(lin)}"

    new_gelu = PACTIntegerGELU(eps_in=eps_in, D=D,export_node=export_node)

    return new_gelu

def integerize_hardswish_fun(gm : fx.GraphModule, match : Match, D=2**14, export_node = False):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    lin_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    assert isinstance(module, PACTHardswish), f"integerize_hardswish_fun got bad match - expected PACTHardswish, got {type(module)}"

    new_gelu = PACTIntegerHardswish(eps_in=eps_in, eps_s=module.eps_s, export_node=export_node)

    return new_gelu

class IntegerizeConstWrapPass(ModularizePass):
    @staticmethod
    def constwrap_replacement_fn(node):
        module = dict(node.graph._owning_module.named_modules())[node.target]
        return (PACTIntegerConstWrap(), node.args, node.kwargs)

    def __init__(self, **kwargs):
        target = [PACTConstWrap()]
        super().__init__(op='call_module', target=tuple(target), replacement_fn = self.constwrap_replacement_fn, name="CONSTWRAP_REPLACEMENT_PASS")

class IntegerizeMeanPass(ModularizePass):

    @staticmethod
    def mean_replacement_fn(node):
        return (PACTIntegerMean(**node.kwargs), node.args, node.kwargs)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        target = [PACTMean()]
        super().__init__(op='call_module', target=tuple(target), replacement_fn = partial(self.mean_replacement_fn), name="MEAN_INTEGERIZE_PASS")

class IntegerizeTrueDivPass(ModularizePass):
    @staticmethod
    def truediv_replacement_fn(node, integer_node=False):
        module = dict(node.graph._owning_module.named_modules())[node.target]
        if module.stable:
            return (PACTTrueIntegerDiv(module.Delta, eps=module.get_eps_div(), eta=module.eta, integer_node=integer_node), node.args, node.kwargs)
        else:
            return (PACTTrueIntegerDiv(module.Delta, eps=0*module.get_eps_div(), eta=module.eta, integer_node=integer_node), node.args, node.kwargs)

    def __init__(self, Delta=2**14, export_div_node = False, **kwargs):
        self.kwargs = kwargs
        target = [PACTDiv(Delta)]
        super().__init__(op='call_module', target=tuple(target), replacement_fn = partial(self.truediv_replacement_fn, integer_node=export_div_node), name="TRUEDIV_REPLACEMENT_PASS")

def integerize_layernorm_fun(gm : fx.GraphModule, match : Match, affine = True, D=2**12, export_node=False):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    layernorm_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    layernorm = matched_modules[0]
    requant = matched_modules[1]
    eps_in = extract_eps(layernorm_node.meta['quant'].eps_in)
    assert isinstance(layernorm, PACTLayerNorm), f"integerize_layernorm_fun got bad match - expected PACTLayerNorm, got {type(module)}"

    maxval = max(requant.max, -requant.min)

    if affine:
        new_weight = layernorm.weight
        new_bias = layernorm.bias
        new_layernorm = PACTIntegerLayerNorm(n_levels=requant.n_levels, eps_in=eps_in, maxval=maxval, weight=new_weight, bias=new_bias, D=D, export_node=export_node)
    else:
        new_layernorm = PACTIntegerLayerNorm(n_levels=requant.n_levels, eps_in=eps_in, maxval=maxval, weight = 1., bias = 0., D=D, export_node=export_node)

    return new_layernorm

class IntegerizeLayerNormPass(SequentialPass):
    def __init__(self, affine = True, D=2**12, export_layernorm_node = False, symbolic_trace: callable = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTLayerNorm(), PACTAsymmetricAct(256, 'max', True, 'relu'))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_layernorm_fun, affine=affine, D=D, export_node=export_layernorm_node), f'_INTEGER_LAYERNORM_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_LAYERNORM_PASS')

def integerize_rmsnorm_fun(gm : fx.GraphModule, match : Match, affine = True, D=2**12, export_node=False):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    rmsnorm_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    rmsnorm = matched_modules[0]
    requant = matched_modules[1]
    eps_in = extract_eps(rmsnorm_node.meta['quant'].eps_in)
    assert isinstance(rmsnorm, PACTRMSNorm), f"integerize_rmsnorm_fun got bad match - expected PACTRMSNorm, got {type(rmsnorm)}"

    maxval = max(requant.max, -requant.min)

    if affine:
        new_weight = rmsnorm.weight
        new_rmsnorm = PACTIntegerRMSNorm(n_levels=requant.n_levels, eps_in=eps_in, maxval=maxval, weight=new_weight, D=D, export_node=export_node)
    else:
        new_weight = torch.ones(rmsnorm.normalized_shape)
        new_rmsnorm = PACTIntegerRMSNorm(n_levels=requant.n_levels, eps_in=eps_in, maxval=maxval, weight = new_weight, D=D, export_node=export_node)

    return new_rmsnorm

class IntegerizeRMSNormPass(SequentialPass):
    def __init__(self, symbolic_trace: callable, affine = True, D=2**12, export_rmsnorm_node = False, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTRMSNorm(1), PACTAsymmetricAct(256, 'max', True, 'relu'))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_rmsnorm_fun, affine=affine, D=D, export_node=export_rmsnorm_node), f'_INTEGER_RMSNORM_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_RMSNORM_PASS')

class IntegerizeSoftmaxPass(SequentialPass):
    def __init__(self, D=2**12, export_softmax_node = False, symbolic_trace: callable = PACT_symbolic_trace, **kwargs):
        passes = []

        pattern = nn.Sequential(PACTSoftmax())
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_softmax_fun, mode='I-BERT', export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))

        pattern = nn.Sequential(PACTITAMax())
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_softmax_fun, mode='ITA', D=D, export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))

        pattern = nn.Sequential(PACTITAPartialMax())
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_softmax_fun, mode='ITA-Partial', D=D, export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_SOFTMAX_PASS')

class IntegerizeGELUPass(SequentialPass):
    def __init__(self, D=2**14, export_gelu_node=False, symbolic_trace: callable = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTGELU())
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_gelu_fun, D=D,export_node=export_gelu_node), f'_INTEGER_GELU_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_GELU_PASS')

class IntegerizeHardswishPass(SequentialPass):
    def __init__(self, export_hardswish_node=False, symbolic_trace: callable = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTHardswish(eps_s=1.0))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(integerize_hardswish_fun, export_node=export_hardswish_node), f'_INTEGER_SILU_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_HARDSWISH_PASS')

def integerize_pact_conv_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    conv_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    eps_in = extract_eps(conv_node.meta['quant'].eps_in)
    conv = matched_modules[0]
    assert isinstance(conv, (PACTConv1d, PACTConv2d)), f"integerize_pact_conv_fun got bad match - expected PACTConv, got {type(conv)}"
    if conv.bias is not None:
        print(f"integerize_pact_conv_fun: WARNING - FOUND CONV LAYER WITH BIAS; MAKE SURE THIS IS INTENDED!\nLAYER NAME: {conv_node.target}")
    conv_type = nn.Conv2d if isinstance(conv, PACTConv2d) else nn.Conv1d
    pm = 'zeros' if conv.padding_mode == 'eps' else conv.padding_mode
    new_conv = conv_type(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=conv.bias is not None,
                         padding_mode=pm)
    try:
        new_conv.weight.data.copy_(conv.weight_int)
    except RuntimeError as e:
        import ipdb; ipdb.set_trace()

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.get_bias_int(eps_in))
    # annotate the new conv with the number of levels
    new_conv.n_levels = conv.n_levels

    return new_conv


class IntegerizePACTConvPass(SequentialPass):
    def __init__(self, symbolic_trace: callable = PACT_symbolic_trace,):
        passes = []
        for i, c in enumerate((PACTConv1d, PACTConv2d)):
            pattern = nn.Sequential(c(1,1,1))
            name = f"_INTEGERIZE_PACT_CONV{i+1}D_PASS"
            passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, integerize_pact_conv_fun, name))
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

    #import IPython; IPython.embed()
    new_lin = nn.Linear(in_features=lin.in_features,
                        out_features=lin.out_features,
                        #bias=(lin.bias is not None))
                        bias=True)

    new_lin.weight.data.copy_(lin.weight_int.round())
    if lin.bias is not None:
        new_bias = lin.get_bias_int(eps_in).round()
        if len(new_bias.shape) == 2:
            new_bias = torch.diagonal(new_bias,0,dim1=-2, dim2=-1)
        new_lin.bias.data.copy_(new_bias)
    else:
        # this is done to avoid the inference of "MatMul" nodes during export.
        # Those nodes do not preserve weight names and our annotation does not
        # work for them.
        new_lin.bias.data.zero_()


    new_lin.n_levels = lin.n_levels

    return new_lin

class IntegerizePACTLinearPass(ReplaceSequentialPatternPass):
    def __init__(self, symbolic_trace: callable = PACT_symbolic_trace):
        pattern = nn.Sequential(PACTLinear(1,1))
        name = "_INTEGERIZE_PACT_LIN_PASS"
        super(IntegerizePACTLinearPass, self).__init__(pattern, symbolic_trace, integerize_pact_linear_fun, name)

def swap_maxpool_act_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    maxpool = matched_modules[0]
    act = matched_modules[1]
    assert isinstance(maxpool, (nn.MaxPool2d)), f"swap_maxpool_act_fun got bad match - expected torch.nn.MaxPool2d, got {type(maxpool)}"
    assert isinstance(act, (PACTUnsignedAct, PACTAsymmetricAct)), f"swap_maxpool_act_fun got bad match - expected 'PACTUnsignedAct' or 'PACTAsymmetricAct', got {type(act)}"
    new_act = deepcopy(act)
    new_pool = deepcopy(maxpool)
    replacement_sequence = nn.Sequential(new_act, new_pool)
    return replacement_sequence

class SwapMaxPoolActPass(SequentialPass):
    def __init__(self, symbolic_trace: callable = PACT_symbolic_trace,):
        passes = []
        # Whenever there is a MaxPool-Act sequence, switch their positions to Act-MaxPool
        for act_name, act_type in [("UNSIGNED_ACT", PACTUnsignedAct), ("SIGNED_ACT", PACTAsymmetricAct)]:
            for mp_name, mp_type in [("MP2D", nn.MaxPool2d)]:
                pattern = nn.Sequential(mp_type(kernel_size=2), act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
                passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, swap_maxpool_act_fun, f"_SWAP_{mp_name}_{act_name}_PASS"))
        super(SwapMaxPoolActPass, self).__init__(*passes, name_prefix="_SWAP_MAXPOOL_ACT_PASS")


def replace_pact_causalconv1d_padconv1d_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    causalconv_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    causalconv = matched_modules[0]
    new_module = nn.Sequential(
        nn.ConstantPad1d(
            padding=(causalconv._PACTCausalConv1d__padding, 0),
            value=0
        ),
        PACTConv1d(
            in_channels=causalconv.in_channels,
            out_channels=causalconv.out_channels,
            kernel_size=causalconv.kernel_size,
            bias=causalconv.bias is not None,
            stride=causalconv.stride,
            padding=0,
            dilation=causalconv.dilation,
            groups=causalconv.groups,
            padding_mode='zeros',
            n_levels=causalconv.n_levels,
            quantize=causalconv.quantize,
            init_clip=causalconv.init_clip,
            learn_clip=causalconv.learn_clip,
            symm_wts=causalconv.symm_wts,
            nb_std=causalconv.nb_std,
            tqt=causalconv.tqt,
            tqt_beta=causalconv.tqt_beta,
            tqt_clip_grad=causalconv.tqt_clip_grad
        )
    )
    new_module[1].weight.data.copy_(causalconv.weight)
    if causalconv.bias is not None:
        new_module[1].bias.data.copy_(causalconv.bias)
    new_module[1].clip_lo = causalconv.clip_lo
    new_module[1].clip_hi = causalconv.clip_hi
    new_module[1].clipping_params = causalconv.clipping_params
    new_module[1].started = causalconv.started
    return new_module


class ReplacePACTCausalConv1DPass(ReplaceSequentialPatternPass):
    def __init__(self, symbolic_trace: callable = PACT_symbolic_trace):
        pattern = nn.Sequential(PACTCausalConv1d(1,1,1))
        name = "_REPLACE_PACT_CAUSALCONV1D_PADCONV1D_PASS"
        super(ReplacePACTCausalConv1DPass, self).__init__(pattern, symbolic_trace, replace_pact_causalconv1d_padconv1d_fun, name)


def bn_act_to_requant_fun(gm : fx.GraphModule, match : Match, D=2**24, cmsis_requant=False, requant_node=False, skip_identity_rqs=True):
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
    if skip_identity_rqs and (eps_in.numel() == eps_out.numel() == 1 and eps_in == eps_out and bn is None):
        return None

    gamma_h = (bn.weight/torch.sqrt(bn.running_var+bn.eps)) if bn is not None else torch.ones_like(eps_in)
    beta_h = bn.bias - bn.running_mean * gamma_h if bn is not None else torch.zeros_like(gamma_h)
    gamma_h *= eps_in
    gamma_h /= eps_out
    beta_h /= eps_out
    if act.rounding and not cmsis_requant:
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

    requant = RequantShift(gamma_h, beta_h, act.n_levels, signed_act, D, cmsis_requant=cmsis_requant, requant_node=requant_node)
    return requant

class IntegerizeBNActPass(SequentialPass):
    def __init__(self, D : float = 2**24, cmsis_requant=False, requant_node=False, skip_identity_rqs=True, symbolic_trace: callable = PACT_symbolic_trace,):
        passes = []
        # replace all combinations of BN + PACT activation with RequantShift layers
        for act_name, act_type in [("UNSIGNED_ACT", PACTUnsignedAct), ("SIGNED_ACT", PACTAsymmetricAct)]:
            for bn_name, bn_type in [("BN1D", nn.BatchNorm1d), ("BN2D", nn.BatchNorm2d), ("BN3D", nn.BatchNorm3d)]:
                pattern = nn.Sequential(bn_type(1), act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
                passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, bn_act_to_requant_fun, f"_INTEGERIZE_{bn_name}_{act_name}_PASS", D=D, cmsis_requant=cmsis_requant, requant_node=requant_node, skip_identity_rqs=skip_identity_rqs))

            #also replace "freestanding" activations AFTER replacing the BN+Act stacks
            pattern = nn.Sequential(act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
            passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, bn_act_to_requant_fun, f"_INTEGERIZE_{act_name}_PASS", D=D, cmsis_requant=cmsis_requant, requant_node=requant_node, skip_identity_rqs=skip_identity_rqs))

        super(IntegerizeBNActPass, self).__init__(*passes, name_prefix="_INTEGERIZE_BN_ACT_PASS")

def conv_bn_act_to_conv_threshold_fun(gm : fx.GraphModule, match : Match, cutie_style_threshs=False):
    modules = dict(gm.named_modules())
    matched_nodes = [n for n in match.nodes_map.values()][-2:0:-1]
    matched_modules = modules_of_match(gm, match)
    assert len(matched_nodes) == len(matched_modules), "conv_bn_act_to_conv_threshold_fun got unexpected non-'call_module' nodes!"
    conv = matched_modules[0]
    conv_node = matched_nodes[0]
    assert isinstance(conv, (nn.Conv1d, nn.Conv2d)), f"conv_bn_act_to_conv_threshold_fun called on incompatible Conv layer {type(conv)}"
    conv_type = nn.Conv1d if isinstance(conv, nn.Conv1d) else nn.Conv2d
    if len(matched_modules) == 2:
        act = matched_modules[1]
        act_node = matched_nodes[1]
        bn = None
    else:
        assert len(matched_modules) == 3, "conv_bn_act_to_conv_threshold_fun expected match of length 1 or 2!"
        act = matched_modules[2]
        act_node = matched_nodes[2]
        bn = matched_modules[1]
        bn_node = matched_nodes[1]
        assert isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)), f"conv_bn_act_to_conv_threshold_fun called on incompatible BN layer {type(bn)}"
    assert isinstance(act, (PACTUnsignedAct, PACTAsymmetricAct)), f"conv_bn_act_to_conv_threshold_fun called on incompatible activation {type(act)}"
    signed_out = not isinstance(act, PACTUnsignedAct)
    prev_node = conv_node.args[0]
    eps_in = extract_eps(act_node.meta['quant'].eps_in).cpu().clone().detach().squeeze() 
    eps_out = act_node.meta['quant'].eps_out.cpu().clone().detach().squeeze()
    prev_eps_out = prev_node.meta['quant'].eps_out

    if conv.bias is None:
        bias = torch.zeros(conv.out_channels)
    else:
        bias = conv.bias.data

    # if the last layer had an unsigned (PACTUnsignedAct) output, we need to
    # compensate for this
    unsigned_indicator = int(not signed_out)
    if prev_node.meta['quant'].signed_out: # if the position of the node in the graph is 1, then it is the first node
        bias_hat = bias
    else:
        bias_add = conv.weight_q.sum(dim=(1,2,3)) if len(conv.weight_q.shape)==4 else conv.weight_q.sum(dim=(1,2))
        bias_add *= prev_eps_out
        bias_hat = bias + bias_add

    beta_hat = (bias_hat - bn.running_mean.data)/torch.sqrt(bn.running_var.data+bn.eps)
    gamma_hat = 1/torch.sqrt(bn.running_var.data+bn.eps)
    if bn.affine:
        beta_hat *= bn.weight.data
        beta_hat += bn.bias.data
        gamma_hat *= bn.weight.data

    thresh_lo = ((-0.5 + unsigned_indicator)*eps_out-beta_hat)/(gamma_hat*eps_in)
    thresh_hi = ((0.5 + unsigned_indicator)*eps_out-beta_hat)/(gamma_hat*eps_in)
    # if some gamma_hats/gammas are negative, the smaller than/larger than relationships are flipped there.
    # the weights in the convolution preceding the BN will be flipped for those channels, so we can simply flip the
    # thresholds. Important: flip BEFORE rounding otherwise they will be off by one :)
    if bn.affine:
        flip_idxs = bn.weight.data < 0
        thresh_hi[flip_idxs] *= -1
        thresh_lo[flip_idxs] *= -1

    thresh_hi = torch.ceil(thresh_hi)
    # CUTIE's upper thresholding condition is:
    # th(x) = 1 if x > thresh_hi
    # so we need to reduce the threshold by 1
    if cutie_style_threshs:
        thresh_hi = thresh_hi - 1
    thresh_lo = torch.ceil(thresh_lo)

    assert torch.all(thresh_lo <= thresh_hi), 'All thresh_lo need to be <= thresh_hi'

    new_conv = conv_type(in_channels=conv.in_channels,
                        out_channels=conv.out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=None,
                        padding_mode='zeros')
    new_conv.weight.data.copy_(conv.weight_int)

    new_conv.n_levels = conv.n_levels

    if conv_type == nn.Conv1d:
        n_d = 1
    else:
        n_d = 2
    threshold_module = ChannelwiseThreshold(thresh_lo, thresh_hi, n_dim=n_d, signed_out=signed_out)
    replacement_sequence = nn.Sequential(new_conv, threshold_module)

    return replacement_sequence


class TernarizeConvBNActPass(SequentialPass):
    def __init__(self, symbolic_trace: callable = PACT_symbolic_trace):
        passes = []
        # integerize Conv layers and replace all combinations of BN + PACT activation with Threshold layers
        for conv_name, conv_type in [('CONV1D', nn.Conv1d), ('CONV2D', nn.Conv2d)]:
            for act_name, act_type in [("UNSIGNED_ACT", PACTUnsignedAct), ("SIGNED_ACT", PACTAsymmetricAct)]:
                for bn_name, bn_type in [("BN1D", nn.BatchNorm1d), ("BN2D", nn.BatchNorm2d)]:
                    pattern = nn.Sequential(conv_type(1,1,1), bn_type(1), act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
                    passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, conv_bn_act_to_conv_threshold_fun, f"_{bn_name}_{act_name}_TO_THRESHOLD_PASS"))

            #also replace "freestanding" activations AFTER replacing the BN+Act stacks
            pattern = nn.Sequential(conv_type(1,1,1), act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, conv_bn_act_to_conv_threshold_fun, f"_{act_name}_TO_THRESHOLD_PASS"))


        super(TernarizeConvBNActPass, self).__init__(*passes, name_prefix="_BN_ACT_TO_THRESHOLD_PASS")

def embedding_integerize_fun(gm : fx.GraphModule, match : Match, **kwargs):
    modules = gm_modules(gm)

    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    n_levels = modules[matched_nodes[0].target].adder.n_levels
    eps_adder = modules[matched_nodes[0].target].adder.acts[0].get_eps()
    bias = modules[matched_nodes[0].target]._parameters['weights']
    maxval = modules[matched_nodes[0].target].maxval
    eps_in = extract_eps(matched_nodes[0].meta['quant'].eps_in)

    new_embedding = PACTIntegerEmbedding(n_levels=n_levels, weight=bias, eps_in=eps_in, eps_adder=eps_adder, maxval=maxval, twoStage=True, **kwargs)

    return new_embedding

# This can be made much more general -- Current workaround
class IntegerizeEmbeddingsPass(SequentialPass):
    def __init__(self, symbolic_trace: callable = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTEmbedding(torch.Tensor((1.,)), init_clip='max', learn_clip=False, act_kind='identity'))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(embedding_integerize_fun, **kwargs), f'_INTEGERIZE_EMBEDDINGS_PASS'))
        super().__init__(*passes, name_prefix='_INTEGERIZE_EMBEDDING_PASS')


class FixChannelNumbersPass(FxPass):

    def __init__(self, word_align : bool=False, compressed : bool=False):
        super(FixChannelNumbersPass, self).__init__()
        self.word_align = word_align
        self.compressed = compressed

    def retarget(self, gm : fx.GraphModule):
        self.visited_nodes = set()

    def fix_conv_channels(self, gm : fx.GraphModule, node : fx.Node, force_out_channels : int, bw_out : int = 0):
        if node.op == 'call_module' and node not in self.visited_nodes:
            module = module_of_node(gm, node)
            if not self.compressed:
                alignment = 32 if self.word_align else 8
            else:
                alignment = 40 if self.word_align else 10
            if isinstance(module, (PACTConv1d, PACTConv2d)):
                conv_levels = module.n_levels
                bw_in = int(np.ceil(np.log2(node.meta['quant'].n_levels_in)))
                bw_w = int(np.ceil(np.log2(conv_levels)))
                min_bw = np.minimum(bw_in, bw_w)
                assert module.groups in [1, module.in_channels], f"fix_conv_channels: Unsupported groups config for conv {module}; {module.groups} groups not supported"
                if min_bw not in [2,4,8]:
                    print(f"modify_convs: minimum bitwidth {min_bw} will not give sensible result")
                channel_multiple = int(np.ceil(alignment/min_bw))
                in_ch = module.in_channels
                out_ch = module.out_channels
                extra_in_channels = (-in_ch) % channel_multiple
                new_in_ch = in_ch+extra_in_channels
                new_groups = module.groups
                if module.groups == in_ch:
                    new_groups = new_in_ch
                new_out_ch = out_ch
                if force_out_channels > 0:
                    new_out_ch = force_out_channels
                new_weights = torch.zeros(tuple([new_out_ch, (new_in_ch//new_groups)]+[k for k in module.kernel_size])).type_as(module.weight.data)
                new_weights[:out_ch, :in_ch, ...] = module.weight.data
                module.weight.data = new_weights
                # [out_ch, 1, 1, 1] for 2d, [out_ch, 1, 1] for 1d
                clip_size = [new_out_ch] + [1] * (len(module.weight.data.shape) -1)

                new_clip_lo = -torch.ones(tuple(clip_size))
                new_clip_hi = torch.ones(tuple(clip_size))
                new_clip_lo[:out_ch] = module.clip_lo.data
                new_clip_hi[:out_ch] = module.clip_hi.data
                module.clip_lo.data = new_clip_lo
                module.clip_hi.data = new_clip_hi

                if module.bias is not None and force_out_channels > 0:
                    new_bias = torch.zeros([new_out_ch]).type_as(module.bias.data)
                    new_bias[:out_ch] = module.bias.data
                    module.bias.data = new_bias

                print(f"Adjusting Conv {node.target}'s channels: {module.in_channels}/{module.out_channels} ==> {new_in_ch}/{new_out_ch}")
                module.in_channels = new_in_ch
                module.out_channels = new_out_ch
                self.visited_nodes.add(node)
                self.fix_conv_channels(gm, node.all_input_nodes[0], new_in_ch)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                n_ch = module.num_features
                new_out_ch = n_ch
                if force_out_channels > 0:
                    new_out_ch = force_out_channels
                elif bw_out > 0:
                    extra_out_channels = (-n_ch) % int(np.ceil(alignment/bw_out))
                    new_out_ch = n_ch + extra_out_channels
                if new_out_ch > n_ch:
                    def pad_bn_param(t : torch.Tensor, val : float):
                        new_param = torch.full([new_out_ch], val).type_as(module.bias.data)
                        new_param[:n_ch] = t
                        return new_param
                    module.bias.data = pad_bn_param(module.bias.data, 0.)
                    module.weight.data = pad_bn_param(module.weight.data, 1.)
                    module.running_mean = pad_bn_param(module.running_mean, 0.)
                    module.running_var = pad_bn_param(module.running_var, 1.)
                    print(f"Adjusting BN {node.target}'s channels: {module.num_features} ==> {new_out_ch}")
                    module.num_features = new_out_ch
                self.visited_nodes.add(node)
                self.fix_conv_channels(gm, node.all_input_nodes[0], new_out_ch)
            # naively assume that number of channels is propagated through any
            # other module type; if we have a quantized activation at the end of the
            # network, perform output channel alignment
            elif isinstance(module, (_PACTActivation,)):
                self.visited_nodes.add(node)
                for inp in node.all_input_nodes:
                    self.fix_conv_channels(gm, inp, force_out_channels, np.ceil(np.log2(node.meta['quant'].n_levels_out)))
            else:
                self.visited_nodes.add(node)
                for inp in node.all_input_nodes:
                    self.fix_conv_channels(gm, inp, force_out_channels)
        elif node.op != 'placeholder' and node not in self.visited_nodes:
            self.visited_nodes.add(node)
            for inp in node.all_input_nodes:
                    self.fix_conv_channels(gm, inp, force_out_channels)

    def run_pass(self, gm : fx.GraphModule):
        out_nodes = [n for n in gm.graph.nodes if n.op == 'output']
        assert len(out_nodes) == 1, "FixChannelNumbersPass only supports single-output networks!"
        self.fix_conv_channels(gm, out_nodes[0], -1)
        return gm

class SignedToUnsignedInputPass(FxPass):
    #intended to be run on an INTEGERIZED network
    def __init__(self, n_levels_in : int = 256):
        self.n_levels_in = n_levels_in

    def run_pass(self, gm : fx.GraphModule):
        input_nodes = [n for n in gm.graph.nodes if n.op == 'placeholder']
        for in_node in input_nodes:
            users = [n for n in gm.graph.nodes if in_node in n.all_input_nodes]
            user_conv_modules = [module_of_node(u) for u in users if u.op == 'call_module' and isinstance(module_of_node(u), (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
            if len(users) != len(user_conv_modules):
                print(f"SignedToUnsignedInputPass will likely create bogus - input is used by non-module or non-Conv nodes!")


class IntegerizeBNPACTHardActsPass(SequentialPass):
    @staticmethod
    def bn_hardact_qact_to_requant_fun(gm : fx.GraphModule, match : Match, D1=2**19, D2=2**10):
        if not isinstance(D1, torch.Tensor):
            D1 = torch.tensor(D1)
        if not isinstance(D2, torch.Tensor):
            D2 = torch.tensor(D2)

        matched_nodes = get_ordered_active_nodes(match)
        matched_modules = [module_of_node(gm, n) for n in matched_nodes]
        if len(matched_modules) == 2:
            q_act = matched_modules[1]
            q_act_node = matched_nodes[1]
            hard_act = matched_modules[0]
            hard_act_node = matched_nodes[0]
            bn = None
        else:
            assert len(matched_modules) == 3, "bn_hardact_qact_to_requant_fun got wrong number of modules!"
            q_act = matched_modules[2]
            q_act_node = matched_nodes[2]
            hard_act = matched_modules[1]
            hard_act_node = matched_nodes[1]
            bn = matched_modules[0]
            bn_node = matched_nodes[0]

        signed_act = isinstance(q_act, PACTAsymmetricAct)
        is_hardsigmoid = isinstance(hard_act, (PACTHardsigmoid, nn.Hardsigmoid))
        in_node = bn_node if bn else hard_act_node
        eps_in = extract_eps(in_node.meta['quant'].eps_in).cpu().clone().detach().squeeze()
        eps_out = q_act_node.meta['quant'].eps_out.cpu().clone().detach().squeeze()
        eps_s = 1./6

        gamma_h = (bn.weight/torch.sqrt(bn.running_var+bn.eps)) if bn is not None else torch.ones_like(eps_in)
        beta_h = bn.bias - bn.running_mean * gamma_h if bn is not None else torch.zeros_like(gamma_h)
        #
        # the scale of the half epsilon depends on whether we are dealing with
        # a hsigm (scale D1) or a hswish (scale D1^2/D2) activation
        # if we are not dealing with a hardswish, we can perform the addition
        # of bias and 3 in one step
        shift_factor = D1
        if is_hardsigmoid:
            beta_h += 3.
            intermediate_eps = eps_out/eps_s
        else:
            shift_factor = shift_factor * D1/D2
            intermediate_eps = torch.sqrt(eps_out/eps_s)
        # this is where it gets interesting. We perform requantization to
        # sqrt(eps_out/eps_s) scaled by 2^D1 first.
        gamma_h *= D1
        gamma_h *= eps_in / intermediate_eps
        gamma_h = torch.round(gamma_h)
        beta_h *= D1
        beta_h /= intermediate_eps

        # if we want to round, we need to add 1/2 epsilon at the output
        eps_half = None
        if q_act.rounding:
            eps_half = 1./2
            eps_half *= shift_factor
            if is_hardsigmoid:
                beta_h += eps_half

            eps_half = torch.round(eps_half)
        beta_h = torch.round(beta_h)
        # otherwise, 3 will be added separately
        three = torch.tensor(3.)
        three *= D1
        three /= intermediate_eps
        three = torch.round(three)
        # the upper clipping bound (6) needs to be scaled to the same epsilon
        six = torch.tensor(6.)
        six *= D1
        six /= intermediate_eps
        six = torch.round(six)
        # one_over_six = torch.tensor(1./6.)
        # one_over_six /= hard_act.eps_s
        # one_over_six = torch.round(one_over_six)

        #copied str8 out of bn_act_to_requant
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
        #TODO add bias functionality for asymmetric activations
        if signed_act:
            clip_bound = np.floor(q_act.n_levels/2 + 0.01)
            c_lo = torch.tensor(-clip_bound)
            c_hi = torch.tensor(clip_bound)
            if q_act.n_levels % 2 == 0:
                c_hi -= 1.
        else:
            c_lo = torch.tensor(0.)
            c_hi = torch.tensor(float(q_act.n_levels-1))

        # when dealing with a hsigm, the output can be at most 1 - we want to
        # have a node that looks like a RequantShift in the Hsigm case in order
        # to avoid unnecessary operations
        if is_hardsigmoid:
            c_hi = torch.min(c_hi, torch.round(torch.tensor(1./eps_out)))

        #ha_requant = HardActRequantShift(gamma_h, beta_h, three=three,
        #six=six, one_over_six=one_over_six, D1=D1, D2=D2,
        #hsigmoid=is_hardsigmoid, c_lo=c_lo, c_hi=c_hi, eps_half=eps_half)
        ha_requant = HardActRequantShift(gamma_h, beta_h, three=three, six=six, D1=D1, D2=D2, hsigmoid=is_hardsigmoid, c_lo=c_lo, c_hi=c_hi, eps_half=eps_half)

        return ha_requant

    def __init__(self, D1, D2, symbolic_trace: callable = PACT_symbolic_trace):
        passes = []
        for act_name, act_type in [("HARDSIGMOID", PACTHardsigmoid), ("HARDSWISH", PACTHardswish)]:
            for bn_name, bn_type in [("BN1D", nn.BatchNorm1d), ("BN2D", nn.BatchNorm2d), ("BN3D", nn.BatchNorm3d)]:
                pattern = nn.Sequential(bn_type(1), act_type(1.), _PACTActivation(1, 'max', False, 'relu'))
                passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, self.bn_hardact_qact_to_requant_fun, f"_INTEGERIZE_{bn_name}_{act_name}_PASS", D1=D1, D2=D2))
            pattern = nn.Sequential(act_type(1.), _PACTActivation(1, 'max', False, 'relu'))
            passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, self.bn_hardact_qact_to_requant_fun, f"_INTEGERIZE_{act_name}_PASS", D1=D1, D2=D2))

        super(IntegerizeBNPACTHardActsPass, self).__init__(*passes, name_prefix="_INTEGERIZE_BN_HARDACT_QACT_PASS")



class IntegerizePACTNetPass(SequentialPass):
    def __init__(self, shape_in, eps_in : Optional[Union[torch.Tensor, float]] = None, signed_in : bool = True,
                 D : float = 2**24, enable_add_first=False,
                 requant_node=False, n_levels_in : int = 256, fix_channel_numbers=False,
                 convert_input_to_unsigned : bool = False, D1 : float = 2**18, D2 : float = 2**12,
                 ternarize : bool = False, word_align_channels : bool = False,
                 export_layernorm_node = False, export_softmax_node = False,
                 export_gelu_node = False, export_div_node = False, export_rmsnorm_node=False, export_hardswish_node=False,
                 skip_identity_rqs = True, symbolic_trace = PACT_symbolic_trace, verbose=False):

        passes = []
        # start by retracing the network to dissolve any integer ops
        passes.append(RetracePass(symbolic_trace))
        # if there's a MaxPool followed directly by an PACT Activation, swap their positions
        # (will be needed later for the IntegerizeBNActPass)
        passes.append(SwapMaxPoolActPass(symbolic_trace=symbolic_trace))
        # replace all CausalConv1d with a ConstantPad+Conv1d module
        passes.append(ReplacePACTCausalConv1DPass(symbolic_trace=symbolic_trace))
        # SwapMaxPoolActPass and ReplacePACTCausalConv1DPass inserted nn.Sequential modules
        # containing two submodules. Retrace the network again to separate these
        passes.append(RetracePass(symbolic_trace))
        # then run a shape propagation pass so the conversion functions can
        # know what shape a node's output has
        # IMPORTANT: run model.eval() BEFORE running this pass - otherwise the
        # ShapePropPass will contaminate the batchnorm parameters!
        passes.append(ShapePropPass(shape_in))
        # biases of convolutional layers which are not followed by a BN must be
        # folded into a new batchNorm layer and their biases discarded/turned off
        passes.append(InsertBNBetweenBiasedConvAndActsPass())
        #make use of the annotated shapes to disassemble layernorms
        #passes.append(LayerNormDisassemblePass()) first step: merge any
        # convolutions with biases into batch norms
        passes.append(MergeConvBNPass(symbolic_trace))
        # second step: annotate epsilons and n_levels
        passes.append(AnnotateEpsPass(eps_in, n_levels_in=n_levels_in, signed_in=signed_in, verbose=verbose))
        # JUNGVI: RetracePass destroy the activation after the first PACTRMSNorm!!!
        passes.append(IntegerizeRMSNormPass(D=D, symbolic_trace=symbolic_trace, export_rmsnorm_node=export_rmsnorm_node))
        # if desired, insert "ghost channels"
        if fix_channel_numbers:
            passes.append(FixChannelNumbersPass(word_align=word_align_channels, compressed=ternarize))
        # second step: annotate epsilons and n_levels
        #passes.append(AnnotateEpsPass(eps_in, n_levels_in=n_levels_in, verbose=verbose))
        # with epsilons annotated everywhere, we can integerize linear
        # functions (conv and FC)
        if ternarize:
        # Look for Conv-BN-Acts, integerize the Conv and and replace the BN-Act
        # with Threshold layers
            passes.append(TernarizeConvBNActPass(symbolic_trace=symbolic_trace))
        else:
        # simply integerize PACTConvs' convolutional weights
            passes.append(IntegerizePACTConvPass(symbolic_trace=symbolic_trace))
        passes.append(IntegerizePACTLinearPass(symbolic_trace=symbolic_trace))
        #passes.append(IntegerizeBNPACTHardActsPass(D1=D1, D2=D2, symbolic_trace=symbolic_trace))
        passes.append(IntegerizeSoftmaxPass(D=D, symbolic_trace=symbolic_trace, export_softmax_node=export_softmax_node))
        passes.append(IntegerizeLayerNormPass(D=D, symbolic_trace=symbolic_trace, export_layernorm_node=export_layernorm_node))


        passes.append(IntegerizeHardswishPass(symbolic_trace=symbolic_trace, export_hardswish_node=export_hardswish_node))

        passes.append(IntegerizeGELUPass(D=D, symbolic_trace=symbolic_trace, export_gelu_node=export_gelu_node))
        passes.append(IntegerizeBNActPass(D, enable_add_first, symbolic_trace=symbolic_trace, requant_node=requant_node, skip_identity_rqs=skip_identity_rqs))
        passes.append(IntegerizeEmbeddingsPass(cmsis_requant=enable_add_first, symbolic_trace=symbolic_trace, requant_node=requant_node))
        passes.append(IntegerizeTrueDivPass(export_div_node=export_div_node, symbolic_trace=symbolic_trace))
        passes.append(IntegerizeMeanPass())
        passes.append(IntegerizeConstWrapPass())
        passes.append(RQSMergePass())
        super(IntegerizePACTNetPass, self).__init__(*passes, name_prefix="_INTEGERIZE_PACT_NET_PASS")
