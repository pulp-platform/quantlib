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


from typing import Optional, Union
import operator

import torch
from torch import nn, fx
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *

from quantlib.algorithms.pact.pact_functions import AlmostSymmQuantFunc
from .. import FxPass, SequentialPass, InsertModuleBetweenModulesPass, ReplaceSequentialPatternPass
from ...util import gm_modules, get_qualified_prefix
from ...util.tracing import LeafTracer, custom_symbolic_trace



from .pact_util import PACT_symbolic_trace
from .. import FxPass, SequentialPass, InsertModuleBetweenModulesPass, RetracePass

from .pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace

from functools import partial
import copy

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
    def trace_op_trees(node : fx.Node, node_specs : list, cur_tree : Union[None, OpTree], op_trees : list, seen_nodes : Optional[set], always_terminate : bool = False):
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
            module = self.replacement_fn(tree)
            new_target = f"_QL_OP_TREE_REPLACE_{self.name.upper()}{'_' if self.name != '' else ''}{i}"
            gm.add_submodule(new_target, module)
            # add a node for the submodule call
            with gm.graph.inserting_before(tree.end_node):
                new_node = gm.graph.call_module(new_target, args=tree.args)
            # attach the module to the previous users of the tree's end node
            tree.end_node.replace_all_uses_with(new_node)
            # finally, delete the nodes in the tree
            for node in tree.nodes:
                gm.graph.erase_node(node)

        # and we're done...
        return gm

class MatmulReplacementPass(OpTreeReplacementPass):
    matmul_node_specs = [('call_function', (torch.bmm, torch.matmul)),
                      ('call_method', ('matmul',))]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__(node_specs=self.matmul_node_specs, replacement_fn=self.matmul_replacement_fn, name="MATMUL")

    def matmul_replacement_fn(self, tree):
        return PACTIntegerMatmul(**self.kwargs)

    
class AddTreeReplacementPass(OpTreeReplacementPass):
    add_node_specs = [('call_function', (torch.add, operator.add)),
                      ('call_method', ('add',))]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(AddTreeReplacementPass, self).__init__(node_specs=self.add_node_specs, replacement_fn=self.add_replacement_fn, name="ADDITION")


    def add_replacement_fn(self, tree):
        return PACTIntegerAdd(num_args=len(tree.args), act_kind='identity', **self.kwargs)


class ConcatTreeReplacementPass(SequentialPass):
    cat_node_specs = [('call_function', (torch.cat,))]
    stack_node_specs = [('call_function', (torch.stack,))]

    def __init__(self, n_levels : int = 256, init_clip : str = 'max', nb_std : float = 3.):
        self.n_levels = n_levels
        self.init_clip = init_clip
        self.nb_std = nb_std
        passes = []
        passes.append(OpTreeReplacementPass(node_specs=self.cat_node_specs, replacement_fn=self.cat_replacement_fn, name="CONCAT", always_terminate=True))
        passes.append(OpTreeReplacementPass(node_specs=self.stack_node_specs, replacement_fn=self.stack_replacement_fn, name="STACK", always_terminate=True))
        super(ConcatTreeReplacementPass, self).__init__(*passes, name_prefix="_QL_REPLACE_CAT_STACK")

    def cat_replacement_fn(self, tree):
        return PACTIntegerConcat(num_args=len(tree.args), n_levels=self.n_levels, act_kind='identity', init_clip=self.init_clip, nb_std=self.nb_std, stack_flag=False, **(tree.kwargs))

    def stack_replacement_fn(self, tree):
        return PACTIntegerConcat(num_args=len(tree.args), n_levels=self.n_levels, act_kind='identity', init_clip=self.init_clip, nb_std=self.nb_std, stack_flag=True, **(tree.kwargs))

class InsertActivationsBetweenLinearsPass(InsertModuleBetweenModulesPass):
    before_modules = (nn.Conv1d,
                      nn.Conv2d,
                      nn.Conv3d,
                      nn.BatchNorm1d,
                      nn.BatchNorm2d,
                      nn.BatchNorm3d,
                      nn.Linear)
    after_modules = (nn.Conv1d,
                      nn.Conv2d,
                      nn.Conv3d,
                      nn.Linear,
                     PACTIntegerMatmul)

    def __init__(self, signed : bool = True, **kwargs):
        name = "PACT_LINEAR_ACTIVATIONS"
        self.signed = signed
        self.kwargs = kwargs
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

class HarmonizePACTNetPass(SequentialPass):
    def __init__(self, **kwargs):
        passes = []
        passes.append(RetracePass(PACT_symbolic_trace))
        passes.append(AddTreeReplacementPass(**kwargs))
        actpass_kwargs = {k:v for k,v in kwargs.items() if k != 'force_out_eps'}
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
    def __init__(self, **kwargs):
        passes = []
        pattern = nn.Sequential(PACTLayerNorm(256))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, disassemble_layernorm_fun, f'_LAYERNORM_DISASSEMBLE_PASS'))
        super().__init__(*passes, name_prefix='_LAYERNORM_DISASSEMBLE_PASS')

def apply_pass_to_wrap_module(gm : fx.GraphModule, match : Match, _pass = None, _tracer = None):

    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    wrap_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    wrap_module = matched_modules[0]
    n_levels = wrap_module.n_levels
    assert isinstance(wrap_module, PACTWrapModule), f"_replacement_fun got bad match - expected LayerNorm, got {type()}"

    cloneModule = copy.deepcopy(wrap_module.module)
    fx_graph = _tracer.trace(cloneModule)
    fx_model = fx.GraphModule(_tracer.root, fx_graph, _tracer.root.__class__.__name__)
    
    fx_model = _pass.apply(fx_model)
    node = PACTWrapModule(fx_model, n_levels, wrap_module._dict)

    return node

    
class ApplyPassToWrapModule(SequentialPass):
    def __init__(self, _pass, name=''):
        passes = []
        pattern = nn.Sequential(PACTWrapModule(nn.Identity(), n_levels=256))
        
        tracer = LeafTracer(PACT_OPS_INCLUSIVE)
        trace = partial(custom_symbolic_trace, tracer=tracer)
        
        passes.append(ReplaceSequentialPatternPass(pattern, trace, partial(apply_pass_to_wrap_module, _pass=_pass, _tracer=tracer), f'_WRAP_PASS_{name}_subpass'))
        super().__init__(*passes, name_prefix='_WRAP_PASS_{name}_subpass')        

def insert_final_activation(fx_model, n_levels):

    #     SCHEREMO: That's a hack
    node = list(fx_model.graph.nodes.__reversed__())[1]
    totidx = 0
    with fx_model.graph.inserting_after(node):

        new_node_target = f'__QL__WRAPPASS__PACTAct_{totidx}'
        target_prefix = get_qualified_prefix(node.target)
        if type(target_prefix) == str and target_prefix != '':
            new_node_target = '.'.join([target_prefix, new_node_target])

        fx_model.add_submodule(new_node_target, PACTAsymmetricAct(n_levels, leaky=0, act_kind='identity', symm=True))
        #fx_model.add_submodule(new_node_target, qa.pact.PACTUnsignedAct(n_levels, leaky=0, act_kind='relu'))
        new_node = fx_model.graph.call_module(new_node_target, args=tuple([node]))
        totidx = totidx + 1

    for output_node in list(node.users):
        helperList = list(output_node.args)
        for idx, arg in enumerate(helperList):
            if arg == node:
                helperList[idx] = new_node

        output_node.args = tuple(helperList)

    new_node.args = tuple([node])

    # re-route the input of the second linop to the output of the asymmetric quant
    # X ---> Y                X ---> PACTAct ---> Y
    #    |             ====> 
    #    |--> PACTAct

    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model

    
def integerize_wrap_module(gm : fx.GraphModule, match : Match, _pass = None, _tracer = None):

    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    wrap_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    wrap_module = matched_modules[0]
    n_levels = wrap_module.n_levels
    assert isinstance(wrap_module, PACTWrapModule), f"_replacement_fun got bad match - expected LayerNorm, got {type()}"

    # SCHEREMO: Workaround SPECIFICALLY for MultiHead Self-Attention
    eps_in = torch.tensor(wrap_node.meta['quant'].eps_in[0])[0]
    shapes_in = [wrap_node.meta['tensor_meta'].shape]*3
    cloneModule = copy.deepcopy(wrap_module.module)
    
    fx_graph = _tracer.trace(cloneModule)
    fx_model = fx.GraphModule(_tracer.root, fx_graph, _tracer.root.__class__.__name__)

    # fx_model = insert_final_activation(fx_model, n_levels)
    
    IntegerizePass = _pass(shapes_in, eps_in = eps_in)
    fx_model = IntegerizePass.apply(fx_model)
    node = PACTWrapModule(fx_model, n_levels, wrap_module._dict)
    
    return node
    
        
class IntegerizeWrapModules(SequentialPass):
    def __init__(self, _pass, name=''):
        passes = []
        pattern = nn.Sequential(PACTWrapModule(nn.Identity(), 256))
        
        tracer = LeafTracer(PACT_OPS_INCLUSIVE)
        trace = partial(custom_symbolic_trace, tracer=tracer)
        
        passes.append(ReplaceSequentialPatternPass(pattern, trace, partial(integerize_wrap_module, _pass=_pass, _tracer=tracer), f'_WRAP_PASS_INTEGERIZE_subpass'))
        super().__init__(*passes, name_prefix='_WRAP_PASS_INTEGERIZE_subpass')        

def wrap_module_fun(gm : fx.GraphModule, match : Match, wrapClass = None, n_levels=256, _tracer=None, **kwargs):

    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    wrap_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    wrap_module = matched_modules[0]
    assert isinstance(wrap_module, wrapClass), f"_replacement_fun got bad match - expected LayerNorm, got {type()}"
    
    cloneModule = copy.deepcopy(wrap_module)
    
    fx_graph = _tracer.trace(cloneModule)
    fx_model = fx.GraphModule(_tracer.root, fx_graph, _tracer.root.__class__.__name__)

    x_model = insert_final_activation(fx_model, n_levels)

    node = PACTWrapModule(fx_model, n_levels, wrap_module.__dict__)
    
    return node # PACTWrapModule(copy.deepcopy(wrap_module), n_levels)
        
class WrapModulePass(SequentialPass):
    def __init__(self, wrapClass, wrapClassCallable, name = '', n_levels=256, **kwargs):
        passes = []
        pattern = nn.Sequential(wrapClassCallable())
        
        tracer = LeafTracer(PACT_OPS | set([wrapClass]))
        trace = partial(custom_symbolic_trace, tracer=tracer)
        
        passes.append(ReplaceSequentialPatternPass(pattern, trace, partial(wrap_module_fun, wrapClass=wrapClass, n_levels=n_levels, _tracer=tracer), f'_WRAP_{name}_PASS'))
        super().__init__(*passes, name_prefix='_WRAP_{name}_PASS')

def unwrap_module_fun(gm : fx.GraphModule, match : Match, wrapClass = None):

    def reqShiftParams(module):
        return (module.mul, module.add, module.div)
    
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    wrap_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    wrap_module = matched_modules[0]
    assert isinstance(wrap_module, PACTWrapModule), f"_replacement_fun got bad match - expected LayerNorm, got {type()}"

    try:
        dim = wrap_module._dict['dim']
        dim_head = wrap_module._dict['inner_dim']
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
    
    node = PACTWrapMHSA(wq_weight, wq_bias, wq_requant_mul, wq_requant_div,
                        wk_weight, wk_bias, wk_requant_mul, wk_requant_div,
                        wv_weight, wv_bias, wv_requant_mul, wv_requant_div,
                        preattn_requant_mul, preattn_requant_div,
                        postattn_requant_mul, postattn_requant_div,
                        wo_weight, wo_bias, wo_requant_mul, wo_requant_div,
                        dim, heads, dim_head,
                        isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2, n_levels)
    
    return node # PACTWrapModule(copy.deepcopy(wrap_module), n_levels)
        
class UnwrapModulePass(SequentialPass):
    def __init__(self, ReplacementClass, name=''):
        passes = []
        pattern = nn.Sequential(PACTWrapModule(nn.Identity(), 256))
        
        tracer = LeafTracer(PACT_OPS)
        trace = partial(custom_symbolic_trace, tracer=tracer)
        
        passes.append(ReplaceSequentialPatternPass(pattern, trace, partial(unwrap_module_fun, wrapClass=ReplacementClass), f'_UNWRAP_{name}_PASS'))
        super().__init__(*passes, name_prefix='_UNWRAP_{name}_PASS')

