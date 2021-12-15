import copy
from typing import Union, Optional
from dataclasses import dataclass

import numpy as np

import torch
from torch import fx, nn

from quantlib.algorithms.pact.pact_ops import *

from .pass_base import FxPass, ReplaceSequentialPatternPass
from ..util import gm_modules, module_of_node
import math
import operator

__all__ = ['AnnotateEpsPass',
           'extract_eps']

def eps_conversion_pact_linears(m : torch.nn.Module, eps_in : torch.Tensor):
    return m.get_eps_out(eps_in)

def eps_conversion_pact_acts(m : torch.nn.Module, eps_in : torch.Tensor):
    return m.get_eps()

def eps_conversion_invalid(m : torch.nn.Module, *eps_in : torch.Tensor, **kw_eps_in : torch.Tensor):
    assert False, f"Module class: {type(m)} does not have a valid epsilon conversion!"

def eps_conversion_pact_gelu(m : torch.nn.Module, eps_in : torch.Tensor):
    #return (1./(m.n_levels//2-1))
    return torch.Tensor((m.maxval/(m.n_levels//2-1)),)
    #return torch.Tensor(((m.n_levels//2-1)/m.maxval),)

def eps_conversion_matmul(*eps_ins):
    return eps_ins[0] * eps_ins[1]

def eps_conversion_pact_softmax(m : torch.nn.Module, eps_in : torch.Tensor):
    return torch.Tensor((1./(m.n_levels-1.),))
    #return torch.Tensor((m.maxval/(m.n_levels-1),))

def eps_conversion_pact_layernorm(m : torch.nn.Module, eps_in : torch.Tensor):
    return torch.Tensor(max((m.maxval/(m.n_levels//2-1)), 0.),)

def eps_conversion_identity(*eps_ins):
    return eps_ins[0]

def eps_conversion_truediv(*eps_ins, **kwargs):
    import IPython; IPython.embed()
    return eps_ins[0]

# def eps_conversion_mul(*eps_ins):
#     try:
#         return eps_ins[0]*eps_ins[1]
#     except:
#         return eps_ins[0]

def eps_conversion_embedding(m : torch.nn.Module, eps_in : torch.Tensor):
    return m.maxval/(m.adder.n_levels//2-1)

def eps_conversion_PACTWrapModule(m : torch.nn.Module, *eps_in):
    return m.statTracker.get_eps()


#return torch.Tensor((1./m.n_levels,))


_EPS_CONVERSIONS = {PACTLinear : eps_conversion_pact_linears,
                    PACTConv1d : eps_conversion_pact_linears,
                    PACTConv2d : eps_conversion_pact_linears,
                    PACTLinear : eps_conversion_pact_linears,
                    PACTAsymmetricAct : eps_conversion_pact_acts,
                    PACTUnsignedAct : eps_conversion_pact_acts,
                    nn.Conv1d : eps_conversion_invalid,
                    nn.Conv2d : eps_conversion_invalid,
                    nn.Conv3d : eps_conversion_invalid,
                    nn.Linear: eps_conversion_invalid,
                    PACTWrapModule : eps_conversion_PACTWrapModule,
                    PACTEmbedding : eps_conversion_embedding,
                    PACTIntegerGELU : eps_conversion_pact_gelu,
                    PACTIntegerSoftmax : eps_conversion_pact_softmax,
                    PACTGELU : eps_conversion_pact_gelu,
                    PACTSoftmax : eps_conversion_pact_softmax,
                    PACTIntegerLayerNorm : eps_conversion_pact_layernorm,
                    PACTLayerNorm : eps_conversion_pact_layernorm,
                    PACTIntegerMatmul: eps_conversion_matmul,
                    f'_CALL_FUNCTION_{repr(torch.matmul)}' : eps_conversion_matmul,
                    f'_CALL_FUNCTION_{repr(torch.bmm)}' : eps_conversion_matmul,
                    '_CALL_METHOD_view' : eps_conversion_identity,
#                     f'_CALL_FUNCTION_{repr(operator.mul)}' : eps_conversion_mul,
                    f'_CALL_FUNCTION_{repr(operator.truediv)}' : eps_conversion_truediv,
}

# modules which "generate" an eps without needing an input eps
_ORIGINAL_EPS_MODULES = (PACTUnsignedAct, PACTAsymmetricAct)

@dataclass
class QuantInfo:
    eps_in : torch.Tensor
    eps_out : torch.Tensor

class AnnotateEpsPass(FxPass):
    def __init__(self, eps_in : Optional[Union[torch.Tensor, float]]):
        super(AnnotateEpsPass, self).__init__()
        if not isinstance(eps_in, torch.Tensor) and eps_in is not None:
            self.eps_in = torch.tensor(eps_in).reshape(-1)
            self.noeps = False
        elif eps_in is None:
            self.eps_in = torch.tensor(1.0).reshape(1)
            self.noeps = True
        else:
            self.eps_in = eps_in.reshape(-1)
            self.noeps = False

    def run_pass(self, gm : fx.GraphModule):
        modules = gm_modules(gm)
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                node.meta['quant'] = QuantInfo(eps_in=self.eps_in, eps_out=self.eps_in)
                for u in node.users:
                    if self.noeps:
                        assert u.op == 'call_module' and isinstance(module_of_node(gm, u), _ORIGINAL_EPS_MODULES), "If no eps is provided to annotate_eps, all users of placeholder nodes must be in _ORIGINAL_EPS_MODULES!"
                    #u.meta['quant'] = QuantInfo(eps_in=torch.tensor(1.0), eps_out=torch.tensor(-1.0))
            else:
                arg_eps_ins = [i.meta['quant'].eps_out for i in node.args if isinstance(i, fx.Node)]
                other_args = [i for i in node.args if not isinstance(i, fx.Node)]
                kwarg_eps_ins = {k : v.meta['quant'].eps_out for k, v in node.kwargs.items() if isinstance(v, fx.Node)}
                other_kwargs = {k : v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)}
                conversion_kwargs = copy.copy(other_kwargs)
                conversion_kwargs.update(other_kwargs)
                all_eps = arg_eps_ins + [v for v in kwarg_eps_ins.values()]
                eps_in = [arg_eps_ins, kwarg_eps_ins]
                if node.op == 'call_module':
                    m = module_of_node(gm, node)
                    k = type(m)
                    conversion_args = [m] + arg_eps_ins + other_args
                else:
                    assert node.op != 'get_attr', "get_attr nodes are not currently supported!"
                    conversion_args = arg_eps_ins
                    k = f'_{node.op.upper()}_{node.target}'
                try:
                    eps_out = _EPS_CONVERSIONS[k](*conversion_args, **conversion_kwargs)
                except KeyError:
                    print(f"key {k} not found in _EPS_CONVERSIONS!")
                    eps_diffs = [np.abs(e1 - e2) for e1, e2 in zip(all_eps[:-1], all_eps[1:])]
                    try:
                        assert all(d < 1e-8 for d in eps_diffs)
                    except:
                        import IPython; IPython.embed()
                    print(f"Using identity epsilon propagation on node with op {node.op}, target {node.target}!")
                    eps_out = all_eps[0]

                node.meta['quant'] = QuantInfo(eps_in=eps_in, eps_out=eps_out)
        return gm


def extract_eps(eps):
    # eps is an annotation produced by annotate_eps which is assumed to
    # contain only 1 value (node has only 1 input). this function extracts the
    # eps tensor from the list[list, dict].
    e_args = eps[0]
    e_kwargs = [e for e in eps[1].values()]
    e = e_args + e_kwargs
    assert len(e) == 1, f"extract_eps got eps containing multiple values: {eps}"
    return e[0]
