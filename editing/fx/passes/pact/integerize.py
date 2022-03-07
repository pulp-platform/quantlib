#
# general.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
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
from .harmonize import LayerNormDisassemblePass, ApplyPassToWrapModule
from ...util import gm_modules, module_of_node
from ...util.tracing import LeafTracer, custom_symbolic_trace

from quantlib.algorithms.pact.pact_ops import RequantShift

from .pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace

__all__ = ['IntegerizePACTConvPass',
           'IntegerizePACTLinearPass',
           'IntegerizeBNActPass',
           'IntegerizePACTNetPass',
           'IntegerizeSoftmaxPass',
           'IntegerizeGELUPass',
           'IntegerizeLayerNormPass',
           'IntegerizeEmbeddingsPass',
           'FixChannelNumbersPass',
           'PACTTracer',
           'PACT_symbolic_trace',]

def integerize_softmax_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    lin_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    assert isinstance(module, PACTSoftmax), f"integerize_softmax_fun got bad match - expected PACTSoftmax, got {type(module)}"

    new_softmax = PACTIntegerSoftmax(n_levels=module.n_levels, eps_in=eps_in)

    return new_softmax

def integerize_gelu_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    lin_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    assert isinstance(module, PACTGELU), f"integerize_gelu_fun got bad match - expected PACTGELU, got {type(lin)}"

    new_gelu = PACTIntegerGELU(n_levels=module.n_levels, eps_in=eps_in)

    return new_gelu

def integerize_layernorm_fun(gm : fx.GraphModule, match : Match, affine = True):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    layernorm_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(layernorm_node.meta['quant'].eps_in)
    assert isinstance(module, PACTLayerNorm), f"integerize_layernorm_fun got bad match - expected PACTLayerNorm, got {type(module)}"

    if affine:
        new_layernorm = PACTIntegerLayerNorm(n_levels=module.n_levels, eps_in=eps_in, maxval=module.maxval, weight=module.weight, bias=module.bias)
    else:
        new_layernorm = PACTIntegerLayerNorm(n_levels=module.n_levels, eps_in=eps_in, maxval=module.maxval)

    return new_layernorm

class IntegerizeLayerNormPass(SequentialPass):
    def __init__(self, affine = True, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTLayerNorm(256))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_layernorm_fun, affine=affine), f'_INTEGER_LAYERNORM_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_LAYERNORM_PASS')

class IntegerizeSoftmaxPass(SequentialPass):
    def __init__(self, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTSoftmax(256))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, integerize_softmax_fun, f'_INTEGER_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_SOFTMAX_PASS')

class IntegerizeGELUPass(SequentialPass):
    def __init__(self, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTGELU(256))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, integerize_gelu_fun, f'_INTEGER_GELU_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_GELU_PASS')

# class RequantShift(nn.Module):
#     def __init__(self, mul : torch.Tensor, add : torch.Tensor, n_levels : int, signed : bool = False, D : torch.Tensor = torch.tensor(2**24)):
#         super(RequantShift, self).__init__()
#         self.register_buffer('mul', mul)
#         self.register_buffer('add', add)
#         self.register_buffer('div', D)
#         self.signed = signed
#         self.n_levels_out = n_levels

#     def forward(self, x):
#         x = x * self.mul
#         x = x + self.add
#         x = (x/self.div).floor()
#         if not self.signed:
#             x = torch.clip(x, 0., float(self.n_levels_out-1))
#         else:
#             c = np.floor(self.n_levels_out/2+0.001)
#             if self.n_levels_out % 2:
#                 x = torch.clip(x, -c, c)
#             else:
#                 x = torch.clip(x, -c, c-1)
#         return x

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
                         bias=None,
                         padding_mode=conv.padding_mode)
    try:
        new_conv.weight.data.copy_(conv.weight_int)
    except RuntimeError:
        import ipdb; ipdb.set_trace()

    # annotate the new conv with the number of levels
    new_conv.n_levels = conv.n_levels

    return new_conv


class IntegerizePACTConvPass(SequentialPass):
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

    #import IPython; IPython.embed()
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

def bn_act_to_requant_fun(gm : fx.GraphModule, match : Match, D=2**24, cmsis_requant=False, requant_node=False):
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

    requant = RequantShift(gamma_h, beta_h, act.n_levels, signed_act, D, cmsis_requant=cmsis_requant, requant_node=requant_node)
    return requant

class IntegerizeBNActPass(SequentialPass):
    def __init__(self, D : float = 2**24, cmsis_requant=False, requant_node=False):
        passes = []
        # replace all combinations of BN + PACT activation with RequantShift layers
        for act_name, act_type in [("UNSIGNED_ACT", PACTUnsignedAct), ("SIGNED_ACT", PACTAsymmetricAct)]:
            for bn_name, bn_type in [("BN1D", nn.BatchNorm1d), ("BN2D", nn.BatchNorm2d), ("BN3D", nn.BatchNorm3d)]:
                pattern = nn.Sequential(bn_type(1), act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
                passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, bn_act_to_requant_fun, f"_INTEGERIZE_{bn_name}_{act_name}_PASS", D=D, cmsis_requant=cmsis_requant, requant_node=requant_node))

            #also replace "freestanding" activations AFTER replacing the BN+Act stacks
            pattern = nn.Sequential(act_type(n_levels=256, init_clip='max', learn_clip=False, act_kind='identity'))
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, bn_act_to_requant_fun, f"_INTEGERIZE_{act_name}_PASS", D=D, cmsis_requant=cmsis_requant, requant_node=requant_node))

        super(IntegerizeBNActPass, self).__init__(*passes, name_prefix="_INTEGERIZE_BN_ACT_PASS")

def embedding_integerize_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)

    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    n_levels = modules[matched_nodes[0].target].adder.n_levels
    eps_adder = modules[matched_nodes[0].target].adder.acts[0].get_eps()
    bias = modules[matched_nodes[0].target]._parameters['weights']
    maxval = modules[matched_nodes[0].target].maxval
    eps_in = extract_eps(matched_nodes[0].meta['quant'].eps_in)

    new_embedding = PACTIntegerEmbedding(n_levels=n_levels, weight=bias, eps_in=eps_in, eps_adder=eps_adder, maxval=maxval, twoStage=True)

    return new_embedding

# This can be made much more general -- Current workaround
class IntegerizeEmbeddingsPass(SequentialPass):
    def __init__(self, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTEmbedding(torch.Tensor((1.,)), init_clip='max', learn_clip=False, act_kind='identity'))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(embedding_integerize_fun), f'_INTEGERIZE_EMBEDDINGS_PASS'))
        super().__init__(*passes, name_prefix='_INTEGERIZE_EMBEDDING_PASS')


class FixChannelNumbersPass(FxPass):

    def retarget(self, gm : fx.GraphModule):
        self.visited_nodes = set()

    def fix_conv_channels(self, gm : fx.GraphModule, node : fx.Node, force_out_channels : int):
        if node.op == 'call_module' and node not in self.visited_nodes:
            module = module_of_node(gm, node)
            if isinstance(module, (PACTConv1d, PACTConv2d)):
                conv_levels = module.n_levels
                bw_in = int(np.ceil(np.log2(node.meta['quant'].n_levels_in)))
                bw_w = int(np.ceil(np.log2(conv_levels)))
                min_bw = np.minimum(bw_in, bw_w)
                assert module.groups in [1, module.in_channels], f"fix_conv_channels: Unsupported groups config for conv {module}; {module.groups} groups not supported"
                if min_bw not in [2,4,8]:
                    print(f"modify_convs: minimum bitwidth {min_bw} will not give sensible result")
                channel_multiple = int(np.ceil(8/min_bw))
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
                new_weights[:out_ch, :in_ch, :, :] = module.weight.data
                module.weight.data = new_weights
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
                if force_out_channels > 0:
                    n_ch = module.num_features
                    def pad_bn_param(t : torch.Tensor, val : float):
                        new_param = torch.full([force_out_channels], val).type_as(module.bias.data)
                        new_param[:n_ch] = t
                        return new_param
                    module.bias.data = pad_bn_param(module.bias.data, 0.)
                    module.weight.data = pad_bn_param(module.weight.data, 1.)
                    module.running_mean = pad_bn_param(module.running_mean, 0.)
                    module.running_var = pad_bn_param(module.running_var, 1.)
                    print(f"Adjusting BN {node.target}'s channels: {module.num_features} ==> {force_out_channels}")
                    module.num_features = force_out_channels
                self.visited_nodes.add(node)
                self.fix_conv_channels(gm, node.all_input_nodes[0], force_out_channels)
            # naively assume that number of channels is propagated through any
            # other module type
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



class IntegerizePACTNetPass(SequentialPass):
    def __init__(self, shape_in, eps_in : Optional[Union[torch.Tensor, float]] = None, D : float = 2**24, enable_add_first=False, requant_node=False, n_levels_in : int = 256, fix_channel_numbers=False, convert_input_to_unsigned : bool = False):
        passes = []
        # start by retracing the network to dissolve any integer ops
        passes.append(RetracePass(PACT_symbolic_trace))
        # then run a shape propagation pass so the conversion functions can
        # know what shape a node's output has
        # IMPORTANT: run model.eval() BEFORE running this pass - otherwise the
        # ShapePropPass will contaminate the batchnorm parameters!
        passes.append(ShapePropPass(shape_in))
        # make use of the annotated shapes to disassemble layernorms
        # passes.append(LayerNormDisassemblePass())
        # first step: merge any convolutions with biases into batch norms
        passes.append(MergeConvBNPass(PACT_symbolic_trace))
        # second step: annotate epsilons and n_levels
        passes.append(AnnotateEpsPass(eps_in, n_levels_in=n_levels_in))
        # if desired, insert "ghost channels"
        if fix_channel_numbers:
            passes.append(FixChannelNumbersPass())
        # with epsilons annotated everywhere, we can integerize linear
        # functions (conv and FC)
        passes.append(IntegerizePACTConvPass())
        passes.append(IntegerizePACTLinearPass())
        passes.append(IntegerizeSoftmaxPass())
        passes.append(IntegerizeLayerNormPass())
        passes.append(IntegerizeGELUPass())
        passes.append(IntegerizeBNActPass(D, enable_add_first, requant_node=requant_node))
        passes.append(IntegerizeEmbeddingsPass())
        super(IntegerizePACTNetPass, self).__init__(*passes, name_prefix="_INTEGERIZE_PACT_NET_PASS")
