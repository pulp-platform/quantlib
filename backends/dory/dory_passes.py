import operator
from typing import Optional
from copy import deepcopy
import numpy as np
import torch
from torch import nn, fx
from torch.fx.subgraph_rewriter import Match
from torch.nn.modules.utils import _single, _pair, _triple
from torch.onnx.symbolic_helper import parse_args


from quantlib.algorithms.pact import RequantShift
from quantlib.editing.fx.passes import SequentialPass, ReplaceSequentialPatternPass, ShapePropPass
from quantlib.editing.fx.util import get_ordered_active_nodes, module_of_node
from quantlib.editing.fx.passes.pact import PACT_symbolic_trace, OpTree, OpTreeReplacementPass


class AvgPoolWrap(nn.Sequential):
    # the avg pool wrapper is identical to nn.Sequential, we just need it to
    # distinguish wrapped AvgPool nodes
    def __init__(self, *args):
        super(AvgPoolWrap, self).__init__(*args)



class RemoveRedundantGlobalPoolingPass(SequentialPass):
    GLOBAL_POOL_CLS = (nn.AdaptiveAvgPool1d,
                       nn.AdaptiveAvgPool2d,
                       nn.AdaptiveAvgPool3d,
                       nn.AdaptiveMaxPool1d,
                       nn.AdaptiveMaxPool2d,
                       nn.AdaptiveMaxPool3d,)

    # global avg/max pooling layers that don't do anything
    @staticmethod
    def remove_redundant_pool(gm : fx.GraphModule, match : Match):
        n = get_ordered_active_nodes(match)
        m = [module_of_node(gm, node) for node in n]
        if isinstance(m[0], RemoveRedundantGlobalPoolingPass.GLOBAL_POOL_CLS):
            shape_in = n[0].all_input_nodes[0].meta['tensor_meta'].shape
            shape_out = n[0].meta['tensor_meta'].shape
            # strip batch and channel dimensions
            shape_in = shape_in[2:]
            shape_out = shape_out[2:]
            if shape_in == shape_out:
                return None
            # if the pooling layer does something, return a copy of it
            return deepcopy(m[0])


    def __init__(self):
        passes = []
        name = "REMOVE_REDUNDANT_POOLING"
        for p_cls in self.GLOBAL_POOL_CLS:
            pattern = nn.Sequential(p_cls(1))
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, self.remove_redundant_pool, name))

        super(RemoveRedundantGlobalPoolingPass, self).__init__(*passes)

class AlignAvgPoolPass(SequentialPass):
    GLOBAL_AVGPOOL_CLS = (nn.AdaptiveAvgPool1d,
                       nn.AdaptiveAvgPool2d,
                       nn.AdaptiveAvgPool3d)
    AVGPOOL_CLS = (nn.AvgPool1d,
                   nn.AvgPool2d,
                   nn.AvgPool3d)
    # the purpose of this pass is to align the 'mul' attribute of a
    # requantShift node to the total kernel size
    @staticmethod
    def align_and_wrap_avg_pool(gm : fx.GraphModule, match : Match):
        n = get_ordered_active_nodes(match)
        m = [module_of_node(gm, node) for node in n]
        if isinstance(m[0], AlignAvgPoolPass.GLOBAL_AVGPOOL_CLS):
            shape_in = n[0].all_input_nodes[0].meta['tensor_meta'].shape
            shape_out = n[0].meta['tensor_meta'].shape
            # strip batch and channel dimensions
            shape_in = shape_in[2:]
            shape_out = shape_out[2:]
            if shape_in == shape_out:
                return None
            ks_tot = 1
            for i, (in_d, out_d) in enumerate(zip(shape_in, shape_out)):
                assert in_d % out_d == 0, "AlignAvgPoolPass: Non-integer kernel size in dimension {i} - in_dim {in_d}, out_dim {out_d}!"
                ks_tot *= (in_d//out_d)
        elif isinstance(m[0], AlignAvgPoolPass.AVGPOOL_CLS):
            dim = int(m[0].__class__.__name__[-2])
            if dim == 1:
                ks = _single(m[0].kernel_size)
            elif dim == 2:
                ks = _pair(m[0].kernel_size)
            else:
                ks = _triple(m[0].kernel_size)
            ks_tot = 1
            for k in ks:
                ks_tot *= k
        new_avgpool = deepcopy(m[0])
        new_rqs = deepcopy(m[1])
        # this is why we do this pass in the first place: for bit-true
        # inference, the `mul` attribute must be aligned to the total kernel
        # size
        new_rqs.mul.data = torch.round(new_rqs.mul.data/ks_tot)*ks_tot
        return AvgPoolWrap(new_avgpool, new_rqs)

    def __init__(self):
        passes = []
        for ap_cls in self.GLOBAL_AVGPOOL_CLS + self.AVGPOOL_CLS:
            nd = int(ap_cls.__name__[-2])
            name = f"AVGPOOL{nd}D_DORY"
            if 'Global' in ap_cls.__name__:
                name = "GLOBAL_" + name
            pattern = nn.Sequential(ap_cls(1), RequantShift(torch.zeros([]), torch.zeros([]), 256))
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, self.align_and_wrap_avg_pool, name))

        super(AlignAvgPoolPass, self).__init__(*passes)



class DORYAdder(nn.Module):
    class DORYAdderFun(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x1, x2, params):
            # 'params' should contain requantization parameters for both inputs
            # and the outputs in a dict. input 1/2 parameters' keys start with
            # 'in1_'/'in2_', output parameters' keys start with 'out_' expected
            # parameter keys for each requantization are the respective prefix
            # plus 'mul_i', 'add_i', 'shift_i', 'signed_i' and 'n_levels_i' -
            # the 'i' stands for 'integer', it is used in ONNX export (see
            # symbolic function) to indicate that the respective parameter is
            # in fact an integer. In the exported graph, the '_i' suffix is
            # discarded. 
            params_x1 = {k[4:]:torch.tensor(v) for k,v in params.items() if k.startswith('in1')}
            if params_x1['rq_i']:
                x1 = RequantShift.MyRequantShift.forward(None, x1, params_x1['mul_i'], params_x1['add_i'], int(2**params_x1['shift_i']), params_x1['signed_i'], params_x1['n_levels_i'], False)

            params_x2 = {k[4:]:torch.tensor(v) for k,v in params.items() if k.startswith('in2')}
            if params_x2['rq_i']:
                x2 = RequantShift.MyRequantShift.forward(None, x2, params_x2['mul_i'], params_x2['add_i'], int(2**params_x2['shift_i']), params_x2['signed_i'], params_x2['n_levels_i'], False)
            x_sum = x1 + x2

            #if rq_out:
                #x_sum = rq_out(x_sum)
            params_x_sum = {k[4:]:torch.tensor(v) for k,v in params.items() if k.startswith('out')}
            if params_x_sum['rq_i']:
                x_sum = RequantShift.MyRequantShift.forward(None, x_sum, params_x_sum['mul_i'], params_x_sum['add_i'], int(2**params_x_sum['shift_i']), params_x_sum['signed_i'], params_x_sum['n_levels_i'], False)

            return x_sum

        @staticmethod
        def symbolic(g, x1, x2, params):
            # 'in{1/2}_signed' are inferred automatically by DORY
            params = {k:v for k,v in params.items() if k not in ['in1_signed', 'in2_signed']}
            ret = g.op("Add", x1, x2, **params)
            ret.setType(x1.type())
            return ret


    def __init__(self, in1_requant : Optional[nn.Module], in2_requant : Optional[nn.Module], out_requant : Optional[nn.Module], in1_n_levels : int = 256, in2_n_levels : int = 256, out_n_levels : int = 256):
        super(DORYAdder, self).__init__()
        self.in1_requant = in1_requant
        self.in2_requant = in2_requant
        self.out_requant = out_requant
        self.in1_n_levels = in1_n_levels
        self.in2_n_levels = in2_n_levels
        self.out_n_levels = out_n_levels


    def forward(self, x1, x2):
        params = {}
        out_signed_inferred = False
        for module, name in [(self.in1_requant, "in1"), (self.in2_requant, "in2"), (self.out_requant, "out")]:
            if module:
                mul = int(module.mul.item())
                add = int(module.add.item())
                shift = int(np.log2(module.div.item()))
                n_l = int(module.n_levels_out.item())
                requant = 1
                signed = int(module.signed)
                out_signed_inferred |= module.signed
            else:
                mul = 1
                add = 0
                shift = 0
                n_l = 256
                requant = 0
                signed = False

            params[f"{name}_signed_i"] = signed if (name != 'out' or module) else out_signed_inferred
            params[f"{name}_mul_i"] = mul
            params[f"{name}_add_i"] = add
            params[f"{name}_shift_i"] = shift
            params[f"{name}_n_levels_i"] = n_l
            params[f"{name}_rq_i"] = requant

        return self.DORYAdderFun.apply(x1, x2, params)


class DORYReplaceAddersPass(OpTreeReplacementPass):
    add_node_specs = [('call_function', (torch.add, operator.add)),
                      ('call_method', ('add',))]

    mergeable_modules = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)

    requanting_modules = (DORYAdder,)

    # this pass replaces quantized adders with DORY-compatible custom adder
    # nodes. It does this in the following way:
    # 1. look for adder nodes which have RequantShift nodes at their inputs
    # (and possibly at the output)
    # 2. for each input, check if the RequantShift follows a layer that can be
    # merged. If so, that input will be labelled as "do not requantize"
    # 3. replace the adder nodes with a requantizing adder which requantizes
    # the appropriate inputs (and the output if appropriate)
    # 4. delete those requantization nodes which are absorbed into the adder
    @staticmethod
    def get_input_requant(gm : fx.GraphModule, n : fx.Node):
        # if the input to an adder performs requantization by itself (currently
        # only done by DORYAdders), nothing needs to be done
        rq_module = module_of_node(gm, n)
        if isinstance(module_of_node(gm, n), DORYReplaceAddersPass.requanting_modules):
            return (rq_module.out_n_levels, None)
        inp = n.all_input_nodes[0]
        assert inp.op == "call_module", "DORYReplaceAddersPass found a RequantShift node ({n}) with a very strange input node ({inp}) whose op is not 'call_module' but {inp.op}..."
        inp_module = module_of_node(gm, inp)
        rq_module = module_of_node(gm, n)
        # if the input module is merged with the subsequent Requant module by
        # DORY, we don't want to perform the input requantization in the adder.
        # But we want to annotate the adder with the correct `n_levels`.
        if isinstance(inp_module, DORYReplaceAddersPass.mergeable_modules):
            return (int(rq_module.n_levels_out), None)
        return n, module_of_node(gm, n)


    def dory_replace_adder(self, gm : fx.GraphModule, tree : OpTree):
        # DORY only supports 2-input adders for now
        if len(tree.args) < 2:
            print(f"Warning: DORYReplaceAdderPass got a strange adder with <2 ({len(tree.args)}) inputs...")
            return None
        if len(tree.args) > 2:
            print(f"Warning: DORYReplaceAdderPass got a strange adder with >2 ({len(tree.args)}) inputs...")
            return None

        for a in tree.args:
            if not (a.op == "call_module" and isinstance(module_of_node(gm, a), (RequantShift, DORYAdder))):
                return None

        inp_requants = [self.get_input_requant(gm, a) for a in tree.args]
        in_n_levels = [in_rq[0] if in_rq[1] is None else in_rq[1].n_levels_out for in_rq in inp_requants]
        self.in_requant_nodes += [ir[0] for ir in inp_requants if ir[1] is not None]
        out_requant_module = None
        if len(tree.users) == 1 and tree.users[0].op == "call_module" and isinstance(module_of_node(gm, tree.users[0]), RequantShift):
            self.out_requant_nodes.append(tree.users[0])
            out_requant_module = module_of_node(gm, tree.users[0])
            out_n_levels = out_requant_module.n_levels_out
        else:
            out_n_levels = max(in_n_levels)

        return DORYAdder(in1_requant=deepcopy(inp_requants[0][1]), in2_requant=deepcopy(inp_requants[1][1]), out_requant=out_requant_module, in1_n_levels=in_n_levels[0], in2_n_levels=in_n_levels[1], out_n_levels=out_n_levels)


    def __init__(self):
        super(DORYReplaceAddersPass, self).__init__(node_specs=self.add_node_specs,
                                                    replacement_fn=self.dory_replace_adder,
                                                    name="DORY_ADDER",
                                                    always_terminate=True)
        self.in_requant_nodes = []
        self.out_requant_nodes = []

    def run_pass(self, gm : fx.GraphModule):
        gm = super(DORYReplaceAddersPass, self).run_pass(gm)
        def remove_node_and_module(gm : fx.GraphModule, n : fx.Node):
            assert len(n.all_input_nodes) <= 1, "DORYReplaceAddersPass: Can't remove node with multiple inputs!"
            if len(n.all_input_nodes) == 1:
                n.replace_all_uses_with(n.all_input_nodes[0])
            gm.graph.erase_node(n)
            if n.op == 'call_module':
                gm.delete_submodule(n.target)
            else:
                print(f"Warning: Suspicious node {n} with op {n.op} is being deleted...")

        # after replacing the adder with a DORYAdder, remove the requantShift nodes &
        # modules which have been absorbed into the adder
        for n in self.in_requant_nodes + self.out_requant_nodes:
            remove_node_and_module(gm, n)

        return gm

    def retarget(self, gm : fx.GraphModule):
        # reset the lists of nodes to remove
        self.in_requant_nodes = []
        self.out_requant_nodes = []


class DORYHarmonizePass(SequentialPass):
    def __init__(self, in_shape):
        passes = []
        passes.append(DORYReplaceAddersPass())
        passes.append(ShapePropPass(in_shape))
        passes.append(AlignAvgPoolPass())
        passes.append(RemoveRedundantGlobalPoolingPass())
        super(DORYHarmonizePass, self).__init__(*passes)
