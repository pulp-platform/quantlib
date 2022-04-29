from collections import OrderedDict
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List

from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from ...editors.editors import ApplicationPoint, Rewriter
from ...matching import LinearGraphMatcher
from ....graphs import quantlib_fine_symbolic_trace
from ....graphs import FXOPCODE_CALL_MODULE, nnmodule_from_fxnode
from quantlib.algorithms.qmodules import QConv2d
from quantlib.utils import quantlib_err_header


epsqconv2deps_pattern = OrderedDict([
    ('in_tunnel',  EpsTunnel(eps=torch.Tensor([1.0]))),
    ('qconv2d',    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,))),  # each `QConv2d` is a sub-class of `nn.Conv2d`; after finding all these patterns, I will filter them by checking those that are actually `QConv2d`
    ('out_tunnel', EpsTunnel(eps=torch.Tensor([1.0]))),
])


class LinearOpsIntegeriser(Rewriter):

    def __init__(self):
        name = 'LinearOpsIntegeriser'
        super(LinearOpsIntegeriser, self).__init__(name)

        self._matcher = LinearGraphMatcher(symbolic_trace_fn=quantlib_fine_symbolic_trace, pattern_module=nn.Sequential(epsqconv2deps_pattern))

        self._patternname_2_patternnode = {n.target: n for n in filter(lambda n: (n.op in FXOPCODE_CALL_MODULE) and (n.target in epsqconv2deps_pattern.keys()), self._matcher.pattern_gm.graph.nodes)}
        self._in_tunnel_node  = self._patternname_2_patternnode['in_tunnel']
        self._qconv2d_node    = self._patternname_2_patternnode['qconv2d']
        self._out_tunnel_node = self._patternname_2_patternnode['out_tunnel']

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:

        candidate_matches = self._matcher.find(g)
        candidate_matches = list(filter(lambda match: isinstance(nnmodule_from_fxnode(match.nodes_map[self._qconv2d_node], g), QConv2d), candidate_matches))

        aps = [ApplicationPoint(rewriter=self, graph=g, apcore=match.nodes_map) for match in candidate_matches]
        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:

        # Were the application points computed by this `Rewriter`, and on the target `fx.GraphModule`?
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        old_module = nnmodule_from_fxnode(ap.apcore[self._qconv2d_node], g)
        new_module = nn.Conv2d(in_channels=old_module.in_channels,
                               out_channels=old_module.out_channels,
                               kernel_size=old_module.kernel_size,
                               stride=old_module.stride,
                               padding=old_module.padding,
                               dilation=old_module.dilation,
                               groups=old_module.groups,
                               bias=old_module.bias)

        iweight = torch.round(old_module.qweight.data.clone().detach() / old_module.scale.data.clone().detach())
        new_module.weight.data = iweight  # TODO: should I offload the responsibility of computing the true-quantised parameter array to `_QLinear`? Probably yes.
        # Use round, NOT floor: divisions might yield slightly less than the correct unit you are aiming for!
        if new_module.bias is not None:
            new_module.bias.data = old_module.bias.data.clone().detach()

        self._counter += 1
        new_target = '_'.join([self._name.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        query_in_tunnel  = ap.apcore[self._in_tunnel_node]
        query_qconv      = ap.apcore[self._qconv2d_node]
        query_out_tunnel = ap.apcore[self._out_tunnel_node]

        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(query_in_tunnel):
            new_node = g.graph.call_module(new_target, args=(query_in_tunnel,))
        query_out_tunnel.replace_input_with(query_qconv, new_node)

        g.delete_submodule(query_qconv.target)
        g.graph.erase_node(query_qconv)

        nnmodule_from_fxnode(ap.apcore[self._in_tunnel_node],  g).set_eps_out(torch.ones_like(nnmodule_from_fxnode(query_in_tunnel, g)._eps_out))  # TODO: must have the same shape as the input eps
        nnmodule_from_fxnode(ap.apcore[self._out_tunnel_node], g).set_eps_in(torch.ones_like(nnmodule_from_fxnode(query_out_tunnel, g)._eps_in))   # TODO: must have the same shape as the output eps

        return g
