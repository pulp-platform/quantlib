#
# harmonize.py
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
#

from functools import partial
from typing import Union, Dict, List, Set, Callable
import operator
import copy
from copy import deepcopy
import torch
from torch import nn, fx
from torch.fx.graph_module import GraphModule
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.generic.generic_ops import *
from quantlib.algorithms.pact.pact_functions import AlmostSymmQuantFunc
from .. import FxPass, SequentialPass, InsertModuleBetweenModulesPass, InsertModuleAfterNodePass,ReplaceSequentialPatternPass, AnnotateEpsPass
from .. import RetracePass, ModularizePass
from ...util import gm_modules, module_of_node
from ...util.tracing import LeafTracer, custom_symbolic_trace
from .pact_util import PACT_symbolic_trace
from .pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACT_symbolic_trace


class OpTree:

    def __init__(self, end_node : fx.Node):
        self.end_node = end_node
        self.nodes = [end_node]
        self.open_branches = len(end_node.all_input_nodes)
        assert self.open_branches > 0, "Tried to create OpTree with no branches - something is wrong!"
        # assume that order and assignment of args and kwargs does not matter
        # and they are all treated the same.
        self._args = list(end_node._args) #+ [v for v in end_node.kwargs.values()]
        self.kwargs = end_node.kwargs
        # note the users of the final node now - it may get
        # deleted and then end_node.users becomes useless
        self.users = [u for u in end_node.users]

    def add_node(self, node : fx.Node):
        assert node not in self.nodes, "OpTree.add_node(): something went wrong: you tried to add the same node to a tree twice..."
        assert not self.is_terminated, "Tried to add a node to a terminated tree!"
        self.nodes.append(node)
        # we assume that no node in our tree has more than 1 user
        self.open_branches += (len(node.all_input_nodes) - 1)

    def terminate_branch(self):
        # one branch has reached a node which is not part of the tree, so one
        # branch can be subtracted
        assert not self.is_terminated, "Tried to terminate a branch in an already-terminated tree!"
        self.open_branches -= 1

    @property
    def is_terminated(self):
        return self.open_branches == 0

    @property
    def args(self):
        # really ugly list comprehensions:
        # the inputs to the tree is the list of all inputs to all nodes in the
        # tree, except those inputs which are tree nodes themselves.

        #SCHEREMO : This was some heavy monkey coding right here -- Why would you cast kwargs to their values? They could be reordered or whatever else!!!
        all_args = [arg for node in self.nodes for arg in node._input_nodes.keys() if arg not in self.nodes] #+ [v for node in self.nodes for v in node.kwargs.values() if v not in self.nodes]
        # in the case of concat nodes, the arguments are lists or tuples, so we
        # unpack them
        all_args_unpacked = []
        for arg in all_args:
            if isinstance(arg, (list, tuple)):
                all_args_unpacked += [a for a in arg]
            else:
                all_args_unpacked.append(arg)
        return tuple(all_args_unpacked)

class OpTreeReplacementPass(FxPass):

    def __init__(self, node_specs : list, replacement_fn : callable, name : str = '', always_terminate : bool = False):
        super(OpTreeReplacementPass, self).__init__()
        self.node_specs = node_specs
        self.replacement_fn = replacement_fn
        self.name = name
        self.always_terminate = always_terminate

    @staticmethod
    def node_matches_spec(node : fx.Node, node_spec : tuple):
        return node.op == node_spec[0] and node.target in node_spec[1]

    @staticmethod
    def trace_op_trees(node : fx.Node, node_specs : list, cur_tree : Union[None, OpTree], op_trees : list, seen_nodes : set, always_terminate : bool = False):
        if node in seen_nodes:
            # if we have already seen this node, it is either already part of a
            # tree or it will never be, so if we exited a tree, terminate the
            # branch and return
            if cur_tree is not None:
                cur_tree.terminate_branch()
                if cur_tree.is_terminated:
                    op_trees.append(cur_tree)
            return
        seen_nodes.add(node)
        if any(OpTreeReplacementPass.node_matches_spec(node, spec) for spec in node_specs):
            # the current node belongs to a tree
            if cur_tree is not None and (len(node.users) > 1 or always_terminate):
                # there is a branch, so we need to cut the tree and start a new one
                cur_tree.terminate_branch()
                if cur_tree.is_terminated:
                    op_trees.append(cur_tree)
                cur_tree = OpTree(end_node=node)
            elif cur_tree is None:
                cur_tree = OpTree(end_node=node)
            else:
                cur_tree.add_node(node)
        elif cur_tree is not None:
            # we exited a tree => terminate this branch
            cur_tree.terminate_branch()
            if cur_tree.is_terminated:
                op_trees.append(cur_tree)
            cur_tree = None

        # follow the graph upstream
        for inp in node.all_input_nodes:
            OpTreeReplacementPass.trace_op_trees(inp, node_specs, cur_tree, op_trees, seen_nodes, always_terminate)


    def run_pass(self, gm : fx.GraphModule):
        out_node = list(gm.graph.nodes)[-1]
        op_trees = []
        self.trace_op_trees(out_node, self.node_specs, None, op_trees, set(), self.always_terminate)
        # we have the op trees, now replace them with a module

        for i, tree in enumerate(op_trees):
            # then add the submodule
            module = self.replacement_fn(gm, tree)
            if module is not None:
                new_target = f"_QL_OP_TREE_REPLACE_{self.name.upper()}{'_' if self.name != '' else ''}{i}"
                gm.add_submodule(new_target, module)
                # add a node for the submodule call
                with gm.graph.inserting_before(tree.end_node):
                    new_node = gm.graph.call_module(new_target, args=tree.args)
                # attach the module to the previous users of the tree's end node
                tree.end_node.replace_all_uses_with(new_node)
                #TODO add possibility to modify this as the pass is run - for
                #now, we assume the output of the new node has the same "quant"
                #meta properties as the output of the tree that is being
                #replaced
                if hasattr(tree.end_node, "meta") and "quant" in tree.end_node.meta:
                    new_node.meta['quant'] = deepcopy(tree.end_node.meta['quant'])
                # finally, delete the nodes in the tree
                for node in tree.nodes:
                    gm.graph.erase_node(node)

        # and we're done...
        return gm

class TruedivReplacementPass(ModularizePass):

    @staticmethod
    def truediv_replacement_fn(node, Delta):
        return (PACTDiv(Delta), node.args, node.kwargs)

    def __init__(self, Delta=2**14, **kwargs):
        self.kwargs = kwargs
        target = [operator.truediv]
        super().__init__(op='call_function', target=tuple(target), replacement_fn = partial(self.truediv_replacement_fn, Delta=Delta), name="DIV_REPLACEMENT_PASS")

class MeanFunctionReplacementPass(ModularizePass):

    @staticmethod
    def mean_replacement_fn(node):
        return (PACTMean(), node.args, node.kwargs)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        target = [torch.mean]
        super().__init__(op='call_function', target=tuple(target), replacement_fn = partial(self.mean_replacement_fn), name="MEAN_REPLACEMENT_PASS")

class MeanMethodReplacementPass(ModularizePass):

    @staticmethod
    def mean_replacement_fn(node):
        return (PACTMean(), node.args, node.kwargs)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        target = ['mean']
        super().__init__(op='call_method', target=tuple(target), replacement_fn = partial(self.mean_replacement_fn), name="MEAN_REPLACEMENT_PASS")

class MeanReplacementPass(SequentialPass):

    def __init__(self, **kwargs):
        passes = []
        passes.append(MeanMethodReplacementPass())
        passes.append(MeanFunctionReplacementPass())
        super().__init__(*passes, name_prefix="_MEAN_REPLACEMENT_PASS")

class MatmulReplacementPass(ModularizePass):

    @staticmethod
    def matmul_replacement_fn(node):
        return (PACTIntegerMatmul(), node.args, node.kwargs)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        target = [torch.matmul]
        super().__init__(op='call_function', target=tuple(target), replacement_fn = self.matmul_replacement_fn, name="MATMUL_REPLACEMENT_PASS")

class AddTreeReplacementPass(OpTreeReplacementPass):
    add_node_specs = [('call_function', (torch.add, operator.add)),
                      ('call_method', ('add',))]

    def __init__(self, infer_sign : bool = False, infer_outact : bool = False, **kwargs):
        default_kwargs = {'learn_clip': True, 'init_clip': 'max', 'act_kind': 'identity', 'tqt': True}
        # overwrite defaults with supplied kwargs
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        self.infer_sign = infer_sign
        self.infer_outact = infer_outact
        super(AddTreeReplacementPass, self).__init__(node_specs=self.add_node_specs, replacement_fn=self.add_replacement_fn, name="ADDITION")

    def add_replacement_fn(self, gm : fx.GraphModule, tree : OpTree):
        kwargs_to_pass = self.kwargs
        if self.infer_sign:
            signed_out = False
            signed = []
            for n in tree.args:
                try:
                    signed.append(n.meta['quant'].signed_out)
                except KeyError:
                    import ipdb; ipdb.set_trace()
                signed_out = signed_out or signed[-1]
            signed.append(signed_out)
            kwargs_to_pass['signed'] = signed
        if self.infer_outact:
            if len(tree.users) == 1 and isinstance(module_of_node(gm, tree.users[0]), (PACTUnsignedAct, PACTAsymmetricAct)):
                try:
                    n_levels = kwargs_to_pass['n_levels']
                except KeyError:
                    n_levels = 256
                if isinstance(n_levels, int):
                    n_levels = [n_levels, 0]
                else:
                    n_levels[-1] = 0
                kwargs_to_pass['n_levels'] = n_levels


        return PACTIntegerAdd(num_args=len(tree.args),  **kwargs_to_pass)

class MulReplacementPass(OpTreeReplacementPass):
    mul_node_specs = [('call_function', (torch.mul, operator.mul)),
                      ('call_method', ('mul',))]

    def __init__(self):
        super(MulReplacementPass, self).__init__(node_specs=self.mul_node_specs, replacement_fn=self.mul_replacement_fn, name="MULTIPLICATION", always_terminate=True)

    def mul_replacement_fn(self, gm : fx.GraphModule, tree : OpTree):
        # multiply takes no args
        return Multiply()

class ConcatTreeReplacementPass(SequentialPass):
    cat_node_specs = [('call_function', (torch.cat,))]
    stack_node_specs = [('call_function', (torch.stack,))]

    def __init__(self, **kwargs):

        default_kwargs = {"n_levels": 256, "init_clip": "max", "nb_std": 3., "learn_clip": True, "act_kind": "identity"}
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        passes = []
        passes.append(OpTreeReplacementPass(node_specs=self.cat_node_specs, replacement_fn=self.cat_replacement_fn, name="CONCAT", always_terminate=True))
        passes.append(OpTreeReplacementPass(node_specs=self.stack_node_specs, replacement_fn=self.stack_replacement_fn, name="STACK", always_terminate=True))
        super(ConcatTreeReplacementPass, self).__init__(*passes, name_prefix="_QL_REPLACE_CAT_STACK")

    def cat_replacement_fn(self, gm : fx.GraphModule, tree : OpTree):
        return PACTIntegerConcat(num_args=len(tree.args), **self.kwargs, stack_flag=False, **(tree.kwargs))

    def stack_replacement_fn(self, gm : fx.GraphModule, tree : OpTree):
        return PACTIntegerConcat(num_args=len(tree.args), **self.kwargs, stack_flag=True, **(tree.kwargs))

class InsertActivationsBetweenLinearsPass(InsertModuleBetweenModulesPass):
    before_modules = (nn.Conv1d,
                      nn.Conv2d,
                      nn.Conv3d,
                      nn.BatchNorm1d,
                      nn.BatchNorm2d,
                      nn.BatchNorm3d,
                      nn.Linear,
                      Multiply)
    after_modules = (nn.Conv1d,
                     PACTIntegerMatmul,
                     nn.Conv2d,
                     nn.Conv3d,
                     nn.Linear,
                     Multiply)

    def __init__(self, signed : bool = True, **kwargs):
        name = "PACT_LINEAR_ACTIVATIONS"
        self.signed = signed
        default_kwargs = {'learn_clip' : True, 'tqt' : True, 'init_clip' : 'max', 'act_kind' : 'identity'}
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        super(InsertActivationsBetweenLinearsPass, self).__init__(modules_before=self.before_modules,
                                                                  modules_after=self.after_modules,
                                                                  make_module_fn=self.inserted_module,
                                                                  name=name,
                                                                  combine='force')

    def inserted_module(self, *args, **kwargs):
        if self.signed:
            return PACTAsymmetricAct(**self.kwargs)
        else:
            module_kwargs = {k:v for k, v in self.kwargs.items() if k != "symm"}
            return PACTUnsignedAct(**module_kwargs)

class InsertActivationsAfterLinearsPass(SequentialPass):
    before_modules = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Linear,
        PACTLayerNorm,
        PACTRMSNorm,
        PACTIntegerMatmul,
        PACTDiv,
        PACTSoftmax,
        PACTGELU,
    )

    activation_nodes = (
        PACTUnsignedAct,
        PACTAsymmetricAct,
    )

    def __init__(self, signed : bool = True, **kwargs):
        self.name = "PACT_LINEAR_ACTIVATIONS"
        super(InsertActivationsAfterLinearsPass, self).__init__(name_prefix=self.name)

        self.signed = signed
        default_kwargs = {'learn_clip' : True, 'tqt' : True, 'init_clip' : 'max', 'act_kind' : 'identity'}
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def retarget(self, gm : fx.GraphModule):
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)

        passes = []

        modules = dict(gm.named_modules())
        output_node = list(gm.graph.nodes.__reversed__())[0]
        node_list = self.find_unactivated_linops(output_node, modules, self.before_modules , self.activation_nodes, outputQuant=True)

        idx = 0
        for node in node_list:
            new_module = self.inserted_module(node)
            if new_module is not None:
                passes.append(InsertModuleAfterNodePass(node, new_module, f"{self.name.upper()}_{idx}"))
                idx += 1


        super(InsertActivationsAfterLinearsPass, self).setup_passes(passes)

    def find_unactivated_linops(self, output_node: torch.fx.Node, modules: Dict[str, torch.nn.Module], linear_op_nodes: List[torch.nn.Module], act_nodes: List[torch.nn.Module], outputQuant: bool = False) -> List[torch.fx.Node]:

        node = output_node
        linop_list = []
        offenders = []

        while node._prev != output_node:
            if (node.target in modules.keys()) and any(isinstance(modules[node.target], pattern) for pattern in linear_op_nodes):
                linop_list.append(node)
            node = node._prev

        for node in linop_list:
            if not self.has_trailing_acts_or_output(node, modules, linear_op_nodes, act_nodes, outputQuant):
                offenders.append(node)

        return offenders

    def has_trailing_acts_or_output(self, node: torch.fx.Node, modules: Dict[str, torch.nn.Module], linear_op_nodes: List[torch.nn.Module], act_nodes: List[torch.nn.Module], outputQuant: bool = False) -> bool:

        nodeBool = True

        for output_node in node.users.keys():
            if (outputQuant and output_node.op == 'output') or output_node.op == 'placeholder' or (output_node.target in modules.keys() and isinstance(modules[output_node.target], tuple(linear_op_nodes))):
                return False
            elif (output_node.target in modules.keys()) and isinstance(modules[output_node.target], tuple(act_nodes)):
                continue
            else:
                nodeBool &= self.has_trailing_acts_or_output(output_node, modules, linear_op_nodes, act_nodes, outputQuant)

        return nodeBool

    def inserted_module(self, *args, **kwargs):
        if self.signed:
            return PACTAsymmetricAct(**self.kwargs)
        else:
            module_kwargs = {k:v for k, v in self.kwargs.items() if k != "symm"}
            return PACTUnsignedAct(**module_kwargs)

class InsertBNBetweenBiasedConvAndActsPass(InsertModuleBetweenModulesPass):
    before_modules = (PACTConv1d,
                      PACTConv2d)
    after_modules = (PACTUnsignedAct,
                     PACTAsymmetricAct,
                     PACTHardswish,
                     PACTHardsigmoid)
    @staticmethod
    def make_dummy_bn(m : nn.Module, act : nn.Module):
        if m.bias is None:
            return None
        if isinstance(m, PACTConv1d):
            bn = nn.BatchNorm1d(m.out_channels)
        else:
            assert isinstance(m, PACTConv2d), f"InsertBNBetweenBiasedConvAndActsPass: Got bad module - expected PACTConv1/2d, got {type(m)}"
            bn = nn.BatchNorm2d(m.out_channels)
        bn.weight.data.copy_(torch.ones([m.out_channels]))
        bn.bias.data.copy_(m.bias)
        bn.running_mean.copy_(torch.zeros([m.out_channels]))
        bn.running_var.copy_(torch.ones([m.out_channels])-bn.eps)
        # Hacky: we are editing the source module...
        m.bias = None
        return bn

    def __init__(self):
        super(InsertBNBetweenBiasedConvAndActsPass, self).__init__(modules_before=self.before_modules,
                                                                   modules_after=self.after_modules,
                                                                   make_module_fn=self.make_dummy_bn,
                                                                   name="BIASED_CONV_AND_ACT")

class HarmonizePACTNetPass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, **kwargs):
        passes = []

        passes.append(RetracePass(symbolic_trace))
        passes.append(AnnotateEpsPass(eps_in=1.0, n_levels_in=256, signed_in=True, prop_n_levels=False, prop_eps=False))
        passes.append(AddTreeReplacementPass(**kwargs))
        passes.append(MulReplacementPass())
        passes.append(MeanReplacementPass())
        passes.append(MatmulReplacementPass())
        passes.append(TruedivReplacementPass())
        actpass_kwargs = {k:v for k,v in kwargs.items() if k not in ['force_out_eps', 'infer_outact', 'infer_sign']}
        if 'n_levels' in kwargs.keys():
            actpass_kwargs['n_levels'] = kwargs['n_levels'][0] if not isinstance(kwargs['n_levels'], int) else kwargs['n_levels']
        passes.append(InsertActivationsBetweenLinearsPass(signed=True, act_kind='identity', **actpass_kwargs))
        super(HarmonizePACTNetPass, self).__init__(*passes, name_prefix='_HARMONIZE_PACT_NET_PASS')

def disassemble_layernorm_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    layernorm_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    layernorm = matched_modules[0]
    assert isinstance(layernorm, PACTLayerNorm), f"layernorm_replacement_fun got bad match - expected LayerNorm, got {type(layernorm)}"

    weight = layernorm._parameters['weight'].detach().clone()
    bias = layernorm._parameters['bias'].detach().clone()
    try:
        shape = layernorm_node.meta['tensor_meta'].shape
    except:
        print("Could not access shape of layernorm layer - please run ShapePropPass before LayerNormDisassemblePass!")
        exit()

    new_layernorm = torch.nn.LayerNorm(layernorm.normalized_shape, elementwise_affine=False)
    new_layernorm.eval()

    if len(shape) > 3:
        batchnorm = torch.nn.BatchNorm2d(layernorm.normalized_shape[0])
    else:
        batchnorm = torch.nn.BatchNorm1d(layernorm.normalized_shape[0])

    batchnorm._parameters['weight'].data = weight
    batchnorm._parameters['bias'].data = bias
    batchnorm.eval()

    activation = PACTAsymmetricAct(n_levels=layernorm.n_levels, act_kind='identity', init_clip='max', learn_clip=False, leaky=0.0, symm=True)
    activation.clip_hi.data.copy_(-layernorm.maxval)
    activation.clip_lo.data.copy_(AlmostSymmQuantFunc.apply(-layernorm.maxval, layernorm.n_levels))
    activation.eval()

    return torch.nn.Sequential(*[new_layernorm, batchnorm, activation])

class LayerNormDisassemblePass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTLayerNorm(256))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, disassemble_layernorm_fun, f'_LAYERNORM_DISASSEMBLE_PASS'))
        super().__init__(*passes, name_prefix='_LAYERNORM_DISASSEMBLE_PASS')

def rqs_merge_fun(gm : fx.GraphModule, match : Match):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    rqs_1_node = matched_nodes[0]
    rqs_2_node = matched_nodes[1]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    rqs_1 = matched_modules[0]
    rqs_2 = matched_modules[1]

    if not (rqs_1.signed == rqs_2.signed) or not (rqs_1.cmsis_requant == rqs_2.cmsis_requant):
        return torch.nn.Sequential(rqs_1, rqs_2)

    # If mixed, it's unsigned;
    mixed = not (rqs_2.signed == rqs_1.signed)

    signed = (rqs_2.signed and rqs_1.signed)

    newMul = ((rqs_1.mul * rqs_2.mul)/min(rqs_1.div, rqs_2.div)).round()
    newD = max(rqs_2.div, rqs_1.div)

    if not rqs_1.cmsis_requant:
        add1 = rqs_1.add - (0.5*rqs_1.div)
        add2 = rqs_2.add - (0.5*rqs_2.div)
        newAdd = ((add1 * rqs_2.mul + add2 * rqs_1.div)/min(rqs_1.div, rqs_2.div)).round() + newD//2
    else:
        add1 = rqs_1.add
        add2 = rqs_2.add
        newAdd = ((add1 * rqs_2.mul + add2 * rqs_1.div)/min(rqs_1.div, rqs_2.div)).round()

    return RequantShift(newMul, newAdd, (rqs_2.n_levels_out)//(2**mixed), signed = signed, D=newD, cmsis_requant=rqs_1.cmsis_requant, requant_node=rqs_1.requant_node)

class RQSMergePass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(RequantShift(torch.Tensor((1.,)), torch.Tensor((1.,)),1), RequantShift(torch.Tensor((1.,)),torch.Tensor((1.,)),1))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, rqs_merge_fun, f'_RQS_MERGE_PASS'))
        super().__init__(*passes, name_prefix='_RQS_MERGE_PASS')

def apply_wrap_module_fun(node, _pass, _tracer):

    module = dict(node.graph._owning_module.named_modules())[node.target]

    cloneModule = copy.deepcopy(module.module)
    fx_graph = _tracer.trace(cloneModule)
    fx_model = fx.GraphModule(_tracer.root, fx_graph, _tracer.root.__class__.__name__)
    fx_model = _pass.apply(fx_model)

    returnNode = PACTWrapModule(fx_model, module.n_levels, module._dict, module.quantize, **module.actArgs)
    return returnNode, node.args, node.kwargs

class ApplyPassToWrapModule(ModularizePass):
    def __init__(self, _pass, name='', symbolic_nodes: Set[nn.Module] = PACT_OPS_INCLUSIVE):
        pattern = [PACTWrapModule(nn.Identity(), n_levels=256)]
        tracer = LeafTracer(symbolic_nodes)
        trace = partial(custom_symbolic_trace, tracer=tracer)

        super().__init__(op='call_module', target=tuple(pattern), replacement_fn = partial(apply_wrap_module_fun, _pass=_pass, _tracer=tracer), name=f"APPLY_TO_WRAP_PASS_{name}")

def integerize_wrap_module_fun(node, _pass, _tracer, _shape_fun: lambda node: node.meta['tensor_meta'].shape):

    #shape_in = _shape_fun(node)
    shape_in = node.meta['shape_in']
    eps_in = node.meta['quant'].eps_in[0]
    runnablePass = _pass(shape_in, eps_in)
    module = dict(node.graph._owning_module.named_modules())[node.target]

    cloneModule = copy.deepcopy(module.module)
    fx_graph = _tracer.trace(cloneModule)
    fx_model = fx.GraphModule(_tracer.root, fx_graph, _tracer.root.__class__.__name__)
    fx_model = runnablePass.apply(fx_model)
    returnNode = PACTWrapModule(fx_model, module.n_levels, module._dict, False, **module.actArgs)
    return returnNode, node.args, node.kwargs

class IntegerizeWrapModule(ModularizePass):
    def __init__(self, _pass, name='', shape_fun = None):
        pattern = [PACTWrapModule(nn.Identity(), n_levels=256)]
        tracer = LeafTracer(PACT_OPS_INCLUSIVE)
        trace = partial(custom_symbolic_trace, tracer=tracer)
        super().__init__(op='call_module', target=tuple(pattern), replacement_fn = partial(integerize_wrap_module_fun, _pass=_pass, _tracer=tracer, _shape_fun=shape_fun), name=f"APPLY_TO_WRAP_PASS_{name}")

def wrap_module_fun(node, n_levels, quantize, **actArgs):
    module = dict(node.graph._owning_module.named_modules())[node.target]
    returnNode = PACTWrapModule(module, n_levels, module.__dict__, quantize = quantize, **actArgs)
    return returnNode, node.args, node.kwargs

class WrapModulePass(ModularizePass):
    def __init__(self, wrapClass, wrapClassCallable, name = '', n_levels=256, quantize = False, **kwargs):
        self.kwargs = kwargs
        pattern = [wrapClassCallable()]
        super().__init__(op='call_module', target=tuple(pattern), replacement_fn = partial(wrap_module_fun, n_levels=n_levels, **self.kwargs, quantize=quantize), name="WRAP_REPLACEMENT_PASS")

# def unwrap_module_fun(gm : fx.GraphModule, match : Match, wrapClass = None):

#     print("Found match!")

#     def reqShiftParams(module):
#         return (module.mul, module.add, module.div)

#     modules = gm_modules(gm)
#     matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
#     wrap_node = matched_nodes[0]
#     matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
#     wrap_module = matched_modules[0]
#     assert isinstance(wrap_module, PACTWrapModule), f"_replacement_fun got bad match - expected LayerNorm, got {type()}"

#     try:
#         dim = wrap_module._dict['out_dim']
#         dim_head = wrap_module._dict['dim']
#         heads = wrap_module._dict['h']
#     except Exception as e:
#         import IPython; IPython.embed()
#     mod = dict(wrap_module.module.named_parameters())
#     wq_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0.weight']
#     wk_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_1.weight']
#     wv_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_2.weight']
#     wo_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_3.weight']

#     try:
#         wq_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0.bias']
#     except:
#         wq_bias = torch.Tensor((0,))
#     try:
#         wk_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_1.bias']
#     except:
#         wk_bias = torch.Tensor((0,))
#     try:
#         wv_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_2.bias']
#     except:
#         wv_bias = torch.Tensor((0,))
#     try:
#         wo_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_3.bias']
#     except:
#         wo_bias = torch.Tensor((0,))

#     wq_requant_mul, wq_requant_add, wq_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0)
#     wk_requant_mul, wk_requant_add, wk_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_1)
#     wv_requant_mul, wv_requant_add, wv_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_2)
#     preattn_requant_mul, preattn_requant_add, preattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3)
#     postattn_requant_mul, postattn_requant_add, postattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_4)
#     wo_requant_mul, wo_requant_add, wo_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_5)

#     sm = wrap_module.module.AttentionMechanism._QL_REPLACED__INTEGER_SOFTMAX_PASS_0

#     isoftmaxA = sm.coeffA
#     isoftmaxB = sm.coeffB
#     isoftmaxC = sm.coeffC
#     isoftmaxlog2 = sm.log2
#     n_levels = wrap_module.n_levels

#     node = PACTWrapMHSA(wq_weight, wq_bias, wq_requant_mul, wq_requant_div,
#                         wk_weight, wk_bias, wk_requant_mul, wk_requant_div,
#                         wv_weight, wv_bias, wv_requant_mul, wv_requant_div,
#                         preattn_requant_mul, preattn_requant_div,
#                         postattn_requant_mul, postattn_requant_div,
#                         wo_weight, wo_bias, wo_requant_mul, wo_requant_div,
#                         dim, heads, dim_head,
#                         isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2, n_levels)

#     return node # PACTWrapModule(copy.deepcopy(wrap_module), n_levels)

def unwrap_linearattention_fun(wrap_module):
    def reqShiftParams(module):
        return (module.mul, module.add, module.div)

    try:
        dim = wrap_module._dict['out_dim']
        dim_head = wrap_module._dict['dim']
        heads = wrap_module._dict['h']
    except Exception as e:
        import IPython; IPython.embed()
    mod = dict(wrap_module.module.named_parameters())
    wq_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0.weight']
    wk_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_1.weight']
    wv_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_2.weight']
    wo_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_3.weight']

    try:
        wq_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0.bias']
    except:
        wq_bias = torch.Tensor((0,))
    try:
        wk_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_1.bias']
    except:
        wk_bias = torch.Tensor((0,))
    try:
        wv_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_2.bias']
    except:
        wv_bias = torch.Tensor((0,))
    try:
        wo_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_3.bias']
    except:
        wo_bias = torch.Tensor((0,))

    #import IPython; IPython.embed()

    wq_requant_mul, wq_requant_add, wq_requant_div = reqShiftParams(wrap_module.module.AttentionMechanism.kernel_func_1._QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0)
    wk_requant_mul, wk_requant_add, wk_requant_div = reqShiftParams(wrap_module.module.AttentionMechanism.kernel_func_2._QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1)
    wv_requant_mul, wv_requant_add, wv_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0)
    preattn_requant_mul, preattn_requant_add, preattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_1)
    normalizer_requant_mul, normalizer_requant_add, normalizer_requant_div = reqShiftParams(wrap_module.module.AttentionMechanism._QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2)
    postattn_requant_mul, postattn_requant_add, postattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_2)
    wo_requant_mul, wo_requant_add, wo_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3)

    div = wrap_module.module._QL_TRUEDIV_REPLACEMENT_PASS_MODULARIZED_0
    Delta = div.Delta
    eps = div.eps
    act_type = 0

    n_levels = wrap_module.n_levels

    return (wq_weight, wq_bias, wq_requant_mul, wq_requant_div,
            wk_weight, wk_bias, wk_requant_mul, wk_requant_div,
            wv_weight, wv_bias, wv_requant_mul, wv_requant_div,
            preattn_requant_mul, preattn_requant_div,
            normalizer_requant_mul, normalizer_requant_div,
            postattn_requant_mul, postattn_requant_div,
            wo_weight, wo_bias, wo_requant_mul, wo_requant_div,
            dim, heads, dim_head,
            Delta, eps, act_type, n_levels), {}

def unwrap_clca_fun(wrap_module):
    def reqShiftParams(module):
        return (module.mul, module.add, module.div)

    try:
        dim = wrap_module._dict['out_dim']
        dim_head = wrap_module._dict['dim']
        heads = wrap_module._dict['h']
        out_dim = wrap_module._dict['out_dim']
    except Exception as e:
        import IPython; IPython.embed()
    mod = dict(wrap_module.module.named_parameters())
    wq_weight = mod['WQ._QL_REPLACED__INTEGERIZE_PACT_CONV1D_PASS_0.weight']
    wkv_weight = mod['WKV._QL_REPLACED__INTEGERIZE_PACT_CONV1D_PASS_1.weight']
    wo_weight = mod['out._QL_REPLACED__INTEGERIZE_PACT_CONV1D_PASS_2.weight']

    try:
        wq_bias = mod['WQ._QL_REPLACED__INTEGERIZE_PACT_CONV1D_PASS_0.bias']
    except:
        wq_bias = torch.Tensor((0,))
    try:
        wkv_bias = mod['WKV._QL_REPLACED__INTEGERIZE_PACT_CONV1D_PASS_1.bias']
    except:
        wkv_bias = torch.Tensor((0,))
    try:
        wo_bias = mod['out._QL_REPLACED__INTEGERIZE_PACT_CONV1D_PASS_2.bias']
    except:
        wo_bias = torch.Tensor((0,))

    #import IPython; IPython.embed()

    wq_requant_mul, wq_requant_add, wq_requant_div = reqShiftParams(wrap_module.module.WQ._QL_REPLACED__INTEGERIZE_BN1D_UNSIGNED_ACT_PASS_0)
    wk_requant_mul, wk_requant_add, wk_requant_div = reqShiftParams(wrap_module.module.AttentionMechanism._QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0)
    wv_requant_mul, wv_requant_add, wv_requant_div = reqShiftParams(wrap_module.module.WKV._QL_REPLACED__INTEGERIZE_BN1D_SIGNED_ACT_PASS_0)

    kdiv_requant_mul, kdiv_requant_add, kdiv_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0)

    preattn_requant_mul, preattn_requant_add, preattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_1)

    postattn_requant_mul, postattn_requant_add, postattn_requant_div = reqShiftParams(wrap_module.module.AttentionMechanism._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_2)

    wo_requant_mul, wo_requant_add, wo_requant_div = reqShiftParams(wrap_module.module.out._QL_REPLACED__INTEGERIZE_BN1D_SIGNED_ACT_PASS_1)

    div = wrap_module.module._QL_TRUEDIV_REPLACEMENT_PASS_MODULARIZED_1
    Delta = div.Delta
    eps = div.eps
    eta = div.eta
    act_type = 0

    n_levels = wrap_module.n_levels

    return (wq_weight, wq_bias, wq_requant_mul, wq_requant_add, wq_requant_div,
            wkv_weight, wkv_bias, wk_requant_mul, wk_requant_add, wk_requant_div,
            wv_requant_mul, wv_requant_add, wv_requant_div,
            kdiv_requant_mul, kdiv_requant_add, kdiv_requant_div,
            preattn_requant_mul, preattn_requant_add, preattn_requant_div,
            postattn_requant_mul, postattn_requant_add, postattn_requant_div,
            wo_weight, wo_bias, wo_requant_mul, wo_requant_add, wo_requant_div,
            dim, heads, dim_head, out_dim,
            Delta, eps, eta, act_type, n_levels), {}

def unwrap_mhsa_fun(wrap_module):
    def reqShiftParams(module):
        return (module.mul, module.add, module.div)

    try:
        dim = wrap_module._dict['out_dim']
        dim_head = wrap_module._dict['dim']
        heads = wrap_module._dict['h']
    except Exception as e:
        import IPython; IPython.embed()
    mod = dict(wrap_module.module.named_parameters())
    wq_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0.weight']
    wk_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_1.weight']
    wv_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_2.weight']
    wo_weight = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_3.weight']

    try:
        wq_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0.bias']
    except:
        wq_bias = torch.Tensor((0,))
    try:
        wk_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_1.bias']
    except:
        wk_bias = torch.Tensor((0,))
    try:
        wv_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_2.bias']
    except:
        wv_bias = torch.Tensor((0,))
    try:
        wo_bias = mod['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_3.bias']
    except:
        wo_bias = torch.Tensor((0,))

    wq_requant_mul, wq_requant_add, wq_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0)
    wk_requant_mul, wk_requant_add, wk_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_1)
    wv_requant_mul, wv_requant_add, wv_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_2)
    preattn_requant_mul, preattn_requant_add, preattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3)
    postattn_requant_mul, postattn_requant_add, postattn_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_4)
    wo_requant_mul, wo_requant_add, wo_requant_div = reqShiftParams(wrap_module.module._QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_5)

    sm = wrap_module.module.AttentionMechanism._QL_REPLACED__INTEGER_SOFTMAX_PASS_0

    isoftmaxA = sm.coeffA
    isoftmaxB = sm.coeffB
    isoftmaxC = sm.coeffC
    isoftmaxlog2 = sm.log2
    n_levels = wrap_module.n_levels

    return (wq_weight, wq_bias, wq_requant_mul, wq_requant_div,
            wk_weight, wk_bias, wk_requant_mul, wk_requant_div,
            wv_weight, wv_bias, wv_requant_mul, wv_requant_div,
            preattn_requant_mul, preattn_requant_div,
            postattn_requant_mul, postattn_requant_div,
            wo_weight, wo_bias, wo_requant_mul, wo_requant_div,
            dim, heads, dim_head,
            isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2, n_levels, wrap_module), {}

def unwrap_module_fun(node, wrapClass = None, modules = None, unwrapFunction = None):
    wrap_module = modules[node.target]
    assert isinstance(wrap_module, PACTWrapModule), f"_replacement_fun got bad match - expected LayerNorm, got {type()}"

    args, kwargs = unwrapFunction(wrap_module)

    retNode = wrapClass(*args, **kwargs)

    return retNode, node.args, node.kwargs

class UnwrapModulePass(ModularizePass):
    def __init__(self, ReplacementClass, ReplacementFunction = unwrap_mhsa_fun, modules = None, name=''):
        passes = []
        pattern = nn.Sequential(PACTWrapModule(nn.Identity(), 256))
        tracer = LeafTracer(PACT_OPS)
        trace = partial(custom_symbolic_trace, tracer=tracer)
        super().__init__(op='call_module', target=tuple(pattern), replacement_fn = partial(unwrap_module_fun, wrapClass = ReplacementClass, modules=modules, unwrapFunction= ReplacementFunction), name=f"UNWRAP_PASS_{name}")

def conv1d_replacement_fun(gm : fx.GraphModule, match : Match, *args, **kwargs):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    conv = matched_modules[0]
    assert isinstance(conv, nn.Conv1d), f"conv1d_replacement_fun got bad match - expected nn.Conv1d, got {type(conv)}"
    return PACTConv1d.from_conv1d(conv, **kwargs)

def conv2d_replacement_fun(gm : fx.GraphModule, match : Match, *args, **kwargs):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    conv = matched_modules[0]
    assert isinstance(conv, nn.Conv2d), f"conv2d_replacement_fun got bad match - expected nn.Conv2d, got {type(conv)}"
    return PACTConv2d.from_conv2d(conv, **kwargs)

class HomogeneousFakeQuantConvPass(SequentialPass):
    def __init__(self, pactConvConfig: dict, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace):
        passes = []
        patternList = [nn.Sequential(nn.Conv1d(3, 6, 5)),  nn.Sequential(nn.Conv2d(3, 6, 5))]
        passes.append(ReplaceSequentialPatternPass(nn.Sequential(nn.Conv1d(3, 6, 5)),
                                                            symbolic_trace,
                                                            partial(conv1d_replacement_fun, **pactConvConfig),
                                                            f'PACTIFIED_CONV1D'))
        passes.append(ReplaceSequentialPatternPass(nn.Sequential(nn.Conv2d(3, 6, 5)),
                                                            symbolic_trace,
                                                            partial(conv2d_replacement_fun, **pactConvConfig),
                                                            f'PACTIFIED_CONV2D'))
        super().__init__(*passes, name_prefix='')

def linear_replacement_fun(gm : fx.GraphModule, match : Match, *args, **kwargs):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    linear = matched_modules[0]
    assert isinstance(linear, nn.Linear), f"linear_replacement_fun got bad match - expected nn.Linear, got {type(linear)}"
    return PACTLinear.from_linear(linear, **kwargs)

class HomogeneousFakeQuantLinearPass(SequentialPass):
    def __init__(self, pactLinearConfig: dict, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace):
        passes = []
        pattern = nn.Sequential(nn.Linear(42, 42))
        passes.append(ReplaceSequentialPatternPass(pattern,
                                                            symbolic_trace,
                                                            partial(linear_replacement_fun, **pactLinearConfig),
                                                            f'PACTIFIED_LINEAR'))
        super().__init__(*passes, name_prefix='')

class HomogeneousFakeQuantReluPass(SequentialPass):
    def __init__(self, pactActConfig: dict, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace):
        passes = []
        passes.append(ReplaceSequentialPatternPass(nn.Sequential(nn.ReLU()),
                                                            symbolic_trace,
                                                            lambda x,y : PACTUnsignedAct(**pactActConfig),
                                                            f'PACTIFIED_RELU'))
        passes.append(ReplaceSequentialPatternPass(nn.Sequential(nn.ReLU6()),
                                                            symbolic_trace,
                                                            lambda x,y : PACTUnsignedAct(**pactActConfig),
                                                            f'PACTIFIED_RELU6'))        
        super().__init__(*passes, name_prefix='')

class HomogeneousFakeQuantPass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, convArgs: dict = {}, actArgs: dict = {}, linearArgs: dict = {}):
        passes = []
        passes.append(HomogeneousFakeQuantConvPass(convArgs, symbolic_trace=symbolic_trace))
        passes.append(HomogeneousFakeQuantReluPass(actArgs, symbolic_trace=symbolic_trace))
        passes.append(HomogeneousFakeQuantLinearPass(linearArgs, symbolic_trace=symbolic_trace))
        super().__init__(*passes, name_prefix='')