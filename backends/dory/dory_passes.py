import operator
from typing import Optional
from copy import deepcopy
import numpy as np
import torch
from torch import nn, fx
from torch.fx.subgraph_rewriter import Match
from torch.nn.modules.utils import _single, _pair, _triple

from quantlib.algorithms.pact import RequantShift
from quantlib.editing.fx.passes import SequentialPass, ReplaceSequentialPatternPass, ShapePropPass
from quantlib.editing.fx.util import get_ordered_active_nodes, module_of_node
from quantlib.editing.fx.passes.pact import PACT_symbolic_trace, OpTree, OpTreeReplacementPass

class AvgPoolWrap(nn.Sequential):
    # the avg pool wrapper is identical to nn.Sequential, we just need it to
    # distinguish wrapped AvgPool nodes
    def __init__(self, *args):
        super(AvgPoolWrap, self).__init__(*args)

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
        def forward(ctx, x1, rq1, x2, rq2, rq_out):
            if rq1:
                x1 = rq1(x1)
            if rq2:
                x2 = rq2(x2)

            x_sum = x1 + x2

            if rq_out:
                x_sum = rq_out(x_sum)

            return x_sum

        @staticmethod
        def symbolic(g, x1, rq1, x2, rq2, rq_out):

            params = {}
            params_list = []
            for module, name in [(rq1, "in1"), (rq2, "in2"), (rq_out, "out")]:
                if module:
                    mul = int(module.mul.item())
                    add = int(module.add.item())
                    shift = int(np.log2(module.div.item()))
                    n_l = int(module.n_levels_out.item())
                    requant = 1
                else:
                    mul = 1
                    add = 0
                    shift = 0
                    n_l = 256
                    requant = 0

                params[f"{name}_mul_i"] = mul
                params[f"{name}_add_i"] = add
                params[f"{name}_shift_i"] = shift
                params[f"{name}_n_levels_i"] = n_l
                params[f"{name}_rq_i"] = requant
            ret = g.op("Add", x1, x2, **params)
            ret.setType(x1.type())
            return ret


    def __init__(self, in1_requant : Optional[nn.Module], in2_requant : Optional[nn.Module], out_requant : Optional[nn.Module]):
        super(DORYAdder, self).__init__()
        self.in1_requant = in1_requant
        self.in2_requant = in2_requant
        self.out_requant = out_requant

    def forward(self, x1, x2):
        return self.DORYAdderFun.apply(x1, self.in1_requant, x2, self.in2_requant, self.out_requant)
        #return self.DORYAdderFun.forward(None, x1, self.in1_requant, x2, self.in2_requant, self.out_requant)


class DORYReplaceAddersPass(OpTreeReplacementPass):
    add_node_specs = [('call_function', (torch.add, operator.add)),
                      ('call_method', ('add',))]

    mergeable_modules = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)

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
        inp = n.all_input_nodes[0]
        assert inp.op == "call_module", f"DORYReplaceAddersPass found a RequantShift node ({n}) with a very strange input node ({inp}) whose op is not 'call_module' but {inp.op}..."
        inp_module = module_of_node(gm, inp)
        # try:
        # SCHEREMO: This checks on the input node!
        # assert len(n.users) == 1, f"DORYReplaceAddersPass found a RequantShift going into an adder with <1 ({len(inp.users)}) users at node {inp}, checked {len(n.users)}"
        if len(n.users) > 1:
            return (None, None)
        # except AssertionError as e:
        #     import IPython; IPython.embed()
        if isinstance(inp_module, DORYReplaceAddersPass.mergeable_modules):
            return (None, None)
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
            if not (a.op == "call_module" and isinstance(module_of_node(gm, a), RequantShift)):
                return None

        inp_requants = [self.get_input_requant(gm, a) for a in tree.args]
        self.in_requant_nodes += [ir[0] for ir in inp_requants if ir[0] is not None]
        out_requant_module = None
        if len(tree.users) == 1 and tree.users[0].op == "call_module" and isinstance(module_of_node(gm, tree.users[0]), RequantShift):
            self.out_requant_nodes.append(tree.users[0])
            out_requant_module = module_of_node(gm, tree.users[0])

        return DORYAdder(in1_requant=deepcopy(inp_requants[0][1]), in2_requant=deepcopy(inp_requants[1][1]), out_requant=out_requant_module)


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
            assert len(n.all_input_nodes) == 1, "DORYReplaceAddersPass: Can't remove node with multiple inputs!"
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
        passes.append(ShapePropPass(in_shape))
        passes.append(DORYReplaceAddersPass())
        passes.append(AlignAvgPoolPass())
        super(DORYHarmonizePass, self).__init__(*passes)
