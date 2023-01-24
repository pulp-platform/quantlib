

from typing import Optional, Union
import operator

import torch
from torch import nn, fx
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.generic.generic_ops import *


from quantlib.algorithms.pact.pact_functions import AlmostSymmQuantFunc
from .. import FxPass, SequentialPass, InsertModuleBetweenModulesPass, ReplaceSequentialPatternPass, AnnotateEpsPass
from ...util import gm_modules, get_qualified_prefix, module_of_node
from ...util.tracing import LeafTracer, custom_symbolic_trace



from .bb_util import BB_symbolic_trace
from .. import FxPass, SequentialPass, InsertModuleBetweenModulesPass, RetracePass
from ..pact import OpTree, OpTreeReplacementPass, AddTreeReplacementPass, MulReplacementPass


from functools import partial
import copy

class BBAddTreeReplacementPass(OpTreeReplacementPass):
    add_node_specs = [('call_function', (torch.add, operator.add)),
                      ('call_method', ('add',))]

    def __init__(self, infer_sign : bool = False, infer_outact : bool = False, pact_kwargs : dict = {}, bb_kwargs : dict = {}, **kwargs):

        pact_default_kwargs = {'learn_clip': True, 'init_clip': 'max', 'act_kind': 'identity', 'tqt' : True}
        bb_default_kwargs = {'hc_stretch': 1.2, 'hc_T' : 0.5, 'init_clip': 'max', 'act_kind': 'identity', 'learn_clip': False}
        # overwrite defaults with supplied kwargs
        pact_default_kwargs.update(pact_kwargs)
        bb_default_kwargs.update(bb_kwargs)
        self.pact_kwargs = pact_default_kwargs
        self.bb_kwargs = bb_default_kwargs
        self.infer_sign = infer_sign
        self.infer_outact = infer_outact
        self.kwargs = kwargs
        super(BBAddTreeReplacementPass, self).__init__(node_specs=self.add_node_specs, replacement_fn=self.add_replacement_fn, name="ADDITION")

    def add_replacement_fn(self, gm : fx.GraphModule, tree : OpTree):
        pact_kwargs_to_pass = self.pact_kwargs.copy()
        kwargs_to_pass = self.kwargs.copy()
        if self.infer_sign:
            signed_out = False
            signed = []
            for n in tree.args:
                signed.append(n.meta['quant'].signed_out)
                signed_out = signed_out or signed[-1]
            signed.append(signed_out)
            kwargs_to_pass['signed'] = signed
        if self.infer_outact:
            if len(tree.users) == 1 and isinstance(module_of_node(gm, tree.users[0]), (PACTUnsignedAct, PACTAsymmetricAct, BBAct)):
                # if we don't actually need an output activation, just use
                # PACTIntegerAdd!
                try:
                    n_levels = pact_kwargs_to_pass['n_levels']
                except KeyError:
                    n_levels = 256
                if isinstance(n_levels, int):
                    n_levels = [n_levels, 0]
                else:
                    n_levels[-1] = 0
                pact_kwargs_to_pass['n_levels'] = n_levels
                return PACTIntegerAdd(num_args=len(tree.args), **pact_kwargs_to_pass, **kwargs_to_pass)


        kwargs_to_pass = {k : v for k,v in kwargs_to_pass.items() if k not in  ['force_out_eps']}
        return BBIntegerAdd(num_args=len(tree.args),  pact_kwargs=pact_kwargs_to_pass, bb_kwargs=self.bb_kwargs, **kwargs_to_pass)


class InsertBBActivationsBetweenLinearsPass(InsertModuleBetweenModulesPass):
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
        name = "BB_LINEAR_ACTIVATIONS"
        self.signed = signed
        default_kwargs = {'learn_clip' : False, 'init_clip' : 'max', 'act_kind' : 'identity', 'hc_stretch': 1.2, 'hc_T': 0.5}
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        super(InsertBBActivationsBetweenLinearsPass, self).__init__(modules_before=self.before_modules,
                                                                  modules_after=self.after_modules,
                                                                  make_module_fn=self.inserted_module,
                                                                  name=name,
                                                                  combine='force')

    def inserted_module(self, *args, **kwargs):
        return BBAct(signed=self.signed, **self.kwargs)


class HarmonizeBBNetPass(SequentialPass):
    def __init__(self, bb_adders : bool = False, pact_kwargs : dict = {}, bb_kwargs : dict = {}, **kwargs):
        passes = []

        passes.append(RetracePass(BB_symbolic_trace))
        passes.append(AnnotateEpsPass(eps_in=1.0, n_levels_in=256, signed_in=True, prop_n_levels=False, prop_eps=False))
        if bb_adders:
            passes.append(BBAddTreeReplacementPass(pact_kwargs=pact_kwargs, bb_kwargs=bb_kwargs, **kwargs))
        else:
            passes.append(AddTreeReplacementPass(**pact_kwargs, **kwargs))
        passes.append(MulReplacementPass())
        actpass_kwargs = {k:v for k,v in bb_kwargs.items() if k not in ['signed', 'act_kind']}
        passes.append(InsertBBActivationsBetweenLinearsPass(signed=True, act_kind='identity', **actpass_kwargs))
        super(HarmonizeBBNetPass, self).__init__(*passes, name_prefix='_HARMONIZE_BB_NET_PASS')
