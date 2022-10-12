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

from copy import deepcopy

import numpy as np

import torch
from torch import nn, fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from .pass_base import FxPass, SequentialPass, ModifySequentialPatternPass, ModularizePass
from ..util import module_of_node, get_qualified_prefix

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.generic.generic_ops import *

from collections.abc import Iterable

__all__ = ['MergeConvBNPass',
           'ModularizeActivationsPass',
           'RetracePass',
           'InsertModuleAfterNodePass',
           'InsertModuleBetweenNodesPass',
           'InsertModuleBetweenModulesPass',
           'ShapePropPass',
           'CountMACsPass',
           'MemoryUsagePass',
           'CollectPropertiesPass',
           '_MAC_CNT_FNS']


def madd_of_conv2d(insize, c):
    madds_per_pixel = c.kernel_size[0]*c.kernel_size[1]*c.in_channels*c.out_channels//c.groups
    if isinstance(insize, (torch.Size, tuple, list)):
        if len(insize) > 2:
            insize = insize[2:]
        numel = 1
        for el in insize:
            numel *= el
        #numel = 1; [numel := numel * el for el in insize]
    elif isinstance(insize, int):
        numel = insize * insize
    else:
        assert False, f"invalid insize argument passed to madd_of_conv2d: {insize}"
    return madds_per_pixel * numel//(c.stride[0]*c.stride[1])

def madd_of_lin(_, l):
    return l.in_features * l.out_features

def madd_of_mul(insizes, _):
    numel_max = 0
    for s in insizes:
        assert isinstance(s, torch.Size), f"madd_of_mul only supports torch.Size shapes - got {type(s)}"
        # discard batch dimension
        numel_max = max(s[1:].numel(), numel_max)
    return numel_max

_MAC_CNT_FNS = {nn.Conv2d : madd_of_conv2d,
                PACTConv2d : madd_of_conv2d,
                BBConv2d : madd_of_conv2d,
                nn.Linear : madd_of_lin,
                PACTLinear : madd_of_lin,
                BBLinear : madd_of_lin,
                Multiply : madd_of_mul}

def mem_of_linop(l, _, use_bias):
    num_w_el = l.weight.numel()
    w_mem = num_w_el * np.ceil(np.log2((l.n_levels))) / 8
    b_mem = 0
    if use_bias and l.bias is not None:
        num_b_el = l.bias.numel()
        #assumption: we always use 32b biases
        b_mem = 4*num_b_el

    return w_mem+b_mem

def mem_of_act(l, shp, _):
    #THE ASSUMPTION IS THAT BATCH DIMENSION IS ALWAYS INCLUDED
    if isinstance(shp, (tuple, list)):
        shp = torch.Size(shp)
    if isinstance(shp, torch.Size):
        shp = shp[1:]
    else:
        assert False, "mem_of_act expected tuple, list or size for tensor shape specification"

    num_act_el = shp.numel()
    act_mem = num_act_el * np.ceil(np.log2(l.n_levels))/8
    return act_mem



_MEM_CNT_FNS = {
    PACTConv1d : mem_of_linop,
    PACTConv2d : mem_of_linop,
    BBConv2d : mem_of_linop,
    PACTLinear : mem_of_linop,
    BBLinear : mem_of_linop,
    PACTUnsignedAct : mem_of_act,
    PACTAsymmetricAct : mem_of_act,
    BBAct : mem_of_act}

# functions to translate 'call_module' node properties stored in the 'meta'
# dict
# should return a {key1 : str :: value1 : <any>, key2 : str :: value2 : <any>, ...} dict
_PROPERTY_TRANSL_FNS = {
    'tensor_meta': lambda v: {'out_shape': v.shape}
}
def translate_property(k : str, v):
    try:
        translate_fn = _PROPERTY_TRANSL_FNS[k]
    except KeyError:
        # if no translate_fn is in the dict, return the same value
        translate_fn = lambda v_: {k : v_}
    return translate_fn(v)

# functions to extend the property dictionary based on the node being
# annotated. Specifically, this is needed to annotate not only PACT/BBIntegerAdd
# modules with, e.g., the #MACs but also their output activation submodules.
# functions take as arguments the module, the module's name (node.target) and
# the properties of the supplied module. They should return a dict with which
# the top-level property dict will be updated.
def intadd_outact_dict_update(m : nn.Module, target : str, props : dict):
    return {target+'.act_out' : props.copy()}

_PROP_DICT_EXTENSION_FNS = {
    PACTIntegerAdd : intadd_outact_dict_update,
    BBIntegerAdd : intadd_outact_dict_update
}

def extend_properties(m : nn.Module, target : str, props : dict):
    try:
        extension_fn = _PROP_DICT_EXTENSION_FNS[type(m)]
    except KeyError:
        extension_fn = lambda a, b, c : {}

    return extension_fn(m, target, props)


def merge_conv_bn_fun(ml : list):
    assert len(ml) == 2, "List passed to merge_conv_bn_fun should have length 2"
    conv_module = [m for m in ml if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d))][0]
    bn_module = [m for m in ml if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))][0]
    if conv_module.bias is None:
        return
    bias_data = conv_module.bias.data.clone().detach()
    conv_module.bias = None
    bn_module.running_mean.data -= bias_data



class MergeConvBNPass(SequentialPass):
    def __init__(self, trace : callable = fx.symbolic_trace):
        passes = []
        for conv, bn, dim in [(nn.Conv1d, nn.BatchNorm1d, "1D"), (nn.Conv2d, nn.BatchNorm2d, "2D"), (nn.Conv3d, nn.BatchNorm3d, "3D")]:
            pattern = nn.Sequential(conv(1,1,1), bn(1))
            passes.append(ModifySequentialPatternPass(pattern, trace, merge_conv_bn_fun, f"_MERGE_CONV_BN_{dim}"))
        super(MergeConvBNPass, self).__init__(*passes, name_prefix="_MERGE_CONV_BN_PASS")

class ModularizeActivationsPass(ModularizePass):

    inplace_act_functions = (F.threshold_,
                             F.relu_,
                             F.hardtanh_,
                             F.elu_,
                             F.leaky_relu_,
                             F.rrelu_)

    act_function_to_module = {F.threshold : nn.Threshold,
                              F.threshold_ : nn.Threshold,
                              F.relu : nn.ReLU,
                              F.relu_ : nn.ReLU,
                              F.hardtanh : nn.Hardtanh,
                              F.hardswish : nn.Hardswish,
                              F.relu6 : nn.ReLU6,
                              F.elu : nn.ELU,
                              torch.exp : PACTExp,
                              F.elu_ : nn.ELU,
                              F.selu : nn.SELU,
                              F.celu : nn.CELU,
                              F.leaky_relu : nn.LeakyReLU,
                              F.leaky_relu_ : nn.LeakyReLU,
                              F.prelu : nn.PReLU,
                              F.rrelu : nn.RReLU,
                              F.rrelu_ : nn.RReLU,
                              F.glu : nn.GLU,
                              F.gelu : nn.GELU,
                              F.logsigmoid : nn.LogSigmoid,
                              F.hardshrink : nn.Hardshrink,
                              F.tanhshrink : nn.Tanhshrink,
                              F.softsign : nn.Softsign,
                              F.softplus : nn.Softplus,
                              F.softmin : nn.Softmin,
                              F.softmax : nn.Softmax,
                              F.softshrink : nn.Softshrink,
                              #F.log_softmax : nn.LogSoftmax # interfaces don't
                              #conform as they should for logSoftmax...
                              F.tanh : nn.Tanh,
                              F.sigmoid : nn.Sigmoid,
                              F.hardsigmoid : nn.Hardsigmoid,
                              F.silu : nn.SiLU,
                              F.mish : nn.Mish}


    @staticmethod
    def act_node_to_module(node):
        module_inst_args = node.args[1:]
        module_inst_kwargs = {k:v for k,v in node.kwargs.items() if k != 'input'}
        if node.target in inplace_act_functions:
            module_inst_kwargs['inplace'] = True
        module_call_args = node.args[0:1]
        module_call_kwargs = {k:v for k,v in node.kwargs.items() if k == 'input'}
        module_class = ModularizeActivationsPass.act_function_to_module[node.target]
        return (module_class(*module_inst_args, **module_inst_kwargs), module_call_args, module_call_kwargs)

    def __init__(self):
        super(ModularizeActivationsPass, self).__init__(op='call_function', target=tuple(k for k in self.act_function_to_module.keys()), replacement_fn=self.act_node_to_module, name="MODULARIZE_ACTIVATIONS_PASS")

class RetracePass(FxPass):
    def __init__(self, trace : callable):
        super(RetracePass, self).__init__()
        self.trace = trace

    def run_pass(self, gm : fx.GraphModule):
        return self.trace(gm)

class InsertModuleAfterNodePass(FxPass):
    def __init__(self, node_before : fx.Node, module : nn.Module, name : str, insert_target : Optional[str] = None):
        super(InsertModuleAfterNodePass, self).__init__()
        self.node_before = node_before
        self.module = module
        self.name = name
        self.insert_target = insert_target

    def run_pass(self, gm : fx.GraphModule):
        target_list = []
        if self.insert_target is None and self.node_before.op == 'call_module':
            before_prefix = get_qualified_prefix(self.node_before.target)
            if len(before_prefix):
                target_list.append(before_prefix)
        elif self.insert_target is not None:
            target_list.append(self.insert_target)
        target_list.append(f"_QL_INSERTED_AFTER_{self.name.upper()}")
        new_target = '.'.join(target_list)
        gm.add_submodule(new_target, self.module)

        # need to keep track of users before adding the new node
        users_list = [u for u in self.node_before.users]

        with gm.graph.inserting_after(self.node_before):
            new_node = gm.graph.call_module(new_target, args=(self.node_before,))

        for u in users_list:
            u.replace_input_with(self.node_before, new_node)

        return gm

class InsertModuleBetweenNodesPass(FxPass):
    def __init__(self, node_before : fx.Node, node_after : fx.Node, module : nn.Module, name : str, insert_target : Optional[str] = None):
        super(InsertModuleBetweenNodesPass, self).__init__()
        assert node_after in [u for u in node_before.users], f"Supplied invalid node pair to InsertModuleBetweenNodesPass: Node {node_after} is not a user of {node_before}!"
        self.node_before = node_before
        self.node_after = node_after
        self.module = module
        self.name = name
        self.insert_target = insert_target

    def run_pass(self, gm : fx.GraphModule):
        target_list = []
        if self.insert_target is None and self.node_before.op == 'call_module':
            before_prefix = get_qualified_prefix(self.node_before.target)
            if len(before_prefix):
                target_list.append(before_prefix)
        elif self.insert_target is not None:
            target_list.append(self.insert_target)
        target_list.append(f"_QL_INSERTED_BETWEEN_{self.name.upper()}")
        new_target = '.'.join(target_list)
        gm.add_submodule(new_target, self.module)

        with gm.graph.inserting_after(self.node_before):
            new_node = gm.graph.call_module(new_target, args=(self.node_before,))

        self.node_after.replace_input_with(self.node_before, new_node)

        return gm

class InsertModuleBetweenModulesPass(SequentialPass):
    def __init__(self, modules_before : Union[list, tuple], modules_after : Union[list, tuple], make_module_fn : callable, name : str, combine : Union['force', 'conservative', 'none'] = 'none'):
        # the combine flag specifies what to do with node sequences which match
        # the pattern but where the first node has multiple users:
        # - 'force' will insert a single replacement module after the before_node and before
        # all users, regardless of how many users match the pattern, so long as
        # one user matches
        # - 'conservative' will insert a single replacement module after the before_node
        #  and before all users if all users match the pattern, otherwise it
        # will insert a separate module between the before_node and each of the
        # matching users
        # - 'none' will insert separate modules between the before_node and the
        # matching users regardless of the matching status of the users.
        super(InsertModuleBetweenModulesPass, self).__init__(name_prefix=name)
        self.modules_before = tuple(modules_before)
        self.modules_after = tuple(modules_after)
        self.make_module_fn = make_module_fn
        self.name = name
        self.combine = combine

    def retarget(self, gm : fx.GraphModule):
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)

        passes = []
        idx = 0
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                m = module_of_node(gm, node)
                if isinstance(m, self.modules_before):
                    insert_before_users = []
                    for u in node.users:
                        if u.op == 'call_module':
                            user_module = module_of_node(gm, u)
                            if isinstance(user_module, self.modules_after):
                                insert_before_users.append((u, user_module))
                    if len(insert_before_users) >= 1 and ((len(insert_before_users) == len(node.users) and self.combine == 'conservative') or self.combine == 'force'):
                        new_module = self.make_module_fn(m, insert_before_users[0][1])
                        if new_module is not None:
                            passes.append(InsertModuleAfterNodePass(node, new_module, f"{self.name.upper()}_{idx}"))
                            idx += 1
                    else:
                        for u, user_mod in insert_before_users:
                            new_module = self.make_module_fn(m, user_mod)
                            if new_module is not None:
                                passes.append(InsertModuleBetweenNodesPass(node, u, new_module, f"{self.name.upper()}_{idx}"))
                                idx += 1

        super(InsertModuleBetweenModulesPass, self).setup_passes(passes)

class ShapePropPass(FxPass):
    # a wrapper for the shape propagation pass of torch.fx
    def __init__(self, *shapes_in, dtype_in : torch.dtype = torch.float32):
        super(ShapePropPass, self).__init__()

        # SCHEREMO: Workaround for unpacking multi input shapes
        try:
            self.shapes_in = [torch.Size(s) for s in shapes_in]
        except:
            # This case should ONLY be called if the input is a tuple of a list
            shapes_in = shapes_in[0]
            self.shapes_in = [torch.Size(s) for s in shapes_in]

        self.dtype_in = dtype_in

    def run_pass(self, gm : fx.GraphModule):
        training = gm.training
        gm.eval()
        sp = ShapeProp(gm)
        inp = [torch.rand(s, dtype=self.dtype_in) for s in self.shapes_in]
        try:
            sp.run(*inp)
        except Exception as e:
            import IPython; IPython.embed()
        if training:
            # you really shouldn't be passing over GraphModules in non-eval
            # state, but you do you!
            gm.train()

        for node in gm.graph.nodes:
            in_shps = [n.meta['tensor_meta'].shape for n in node.all_input_nodes if 'tensor_meta' in n.meta.keys()]

            if len(in_shps) == 1:
                node.meta['shape_in'] = in_shps[0]
            else:
                node.meta['shape_in'] = in_shps
        return gm

class CountMACsPass(FxPass):
    # annotate each node for which a counting function is defined with the
    # number of MACs it takes to execute it

    def run_pass(self, gm : fx.GraphModule):
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                m = module_of_node(gm, node)
                k = type(m)
                if k in _MAC_CNT_FNS.keys():
                    shp = node.meta['shape_in']
                    # if len(node.all_input_nodes) != 1:
                    #     print("Multi-input module: double-check result of CountMACsPass")
                    #     in_node = node.all_input_nodes
                    #     shp = [n.meta['tensor_meta'].shape for n in in_node]
                    # else:
                    #     in_node = node.all_input_nodes[0]
                    #     shp = in_node.meta['tensor_meta'].shape
                    node.meta['macs'] = int(_MAC_CNT_FNS[k](shp, m))
        return gm

class MemoryUsagePass(FxPass):
    # annotate each node with the amount of memory the storage of the complete
    # parameters takes
    def run_pass(self, gm: fx.GraphModule):

        for node in gm.graph.nodes:
            if node.op == 'call_module':
                m = module_of_node(gm, node)
                k = type(m)
                if k in _MEM_CNT_FNS.keys():
                    use_bias = False
                    if node.next.op == 'output':
                        use_bias = True
                    shp = node.meta['tensor_meta'].shape
                    node.meta['memory'] = int(np.ceil(_MEM_CNT_FNS[k](m, shp, use_bias)))

        return gm


class CollectPropertiesPass(FxPass):
    def __init__(self, prop_dict : Optional[dict]=None):
        if prop_dict is None:
            prop_dict = {}
        self.prop_dict = prop_dict

    def run_pass(self, gm : fx.GraphModule):
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                m = module_of_node(gm, node)
                pd = {}
                for k, v in node.meta.items():
                    transl_prop = translate_property(k, v)
                    pd.update(transl_prop)

                pd['n_users'] = len(node.users)
                self.prop_dict[node.target] = pd
                self.prop_dict.update(extend_properties(m, node.target, pd))
        gm.prop_dict = deepcopy(self.prop_dict)
        return gm
