from collections import OrderedDict
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List

from ...editors.editors import ApplicationPoint, Rewriter
from ...matching import LinearGraphMatcher
from ....graphs import quantlib_fine_symbolic_trace
from ....graphs import FXOPCODE_CALL_MODULE, nnmodule_from_fxnode
from ....graphs import EpsTunnel, Requant
from quantlib.newalgorithms.qmodules import QReLU
from quantlib.newutils import quantlib_err_header


epsbn2drelueps_pattern = OrderedDict([
    ('in_tunnel',  EpsTunnel(eps=torch.Tensor([1.0]))),
    ('bn2d',       nn.BatchNorm2d(num_features=1)),
    ('qrelu',      nn.ReLU(inplace=True)),
    ('out_tunnel', EpsTunnel(eps=torch.Tensor([1.0]))),
])


class Requantiser(Rewriter):

    def __init__(self, B: int = 24):
        name = 'Requantiser'
        super(Requantiser, self).__init__(name)

        self._D = torch.Tensor([2 ** B])

        self._matcher = LinearGraphMatcher(symbolic_trace_fn=quantlib_fine_symbolic_trace, pattern_module=nn.Sequential(epsbn2drelueps_pattern))

        self._patternname_2_patternnode = {n.target: n for n in filter(lambda n: (n.op in FXOPCODE_CALL_MODULE) and (n.target in epsbn2drelueps_pattern.keys()), self._matcher.pattern_gm.graph.nodes)}
        self._in_tunnel_node  = self._patternname_2_patternnode['in_tunnel']
        self._bn2d_node       = self._patternname_2_patternnode['bn2d']
        self._qrelu_node      = self._patternname_2_patternnode['qrelu']
        self._out_tunnel_node = self._patternname_2_patternnode['out_tunnel']

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:

        candidate_matches = self._matcher.find(g)
        candidate_matches = list(filter(lambda match: isinstance(nnmodule_from_fxnode(match.nodes_map[self._qrelu_node], g), QReLU), candidate_matches))

        aps = [ApplicationPoint(rewriter=self, graph=g, apcore=match.nodes_map) for match in candidate_matches]
        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:

        # Were the application points computed by this `Rewriter`, and on the target `fx.GraphModule`?
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        in_tunnel_module = nnmodule_from_fxnode(ap.apcore[self._in_tunnel_node], g)
        bn2d_module  = nnmodule_from_fxnode(ap.apcore[self._bn2d_node], g)
        qrelu_module = nnmodule_from_fxnode(ap.apcore[self._qrelu_node], g)
        out_tunnel_module = nnmodule_from_fxnode(ap.apcore[self._out_tunnel_node], g)

        shape   = ap.apcore[self._bn2d_node].meta['tensor_meta'].shape
        eps_in  = in_tunnel_module._eps_out
        mi      = bn2d_module.running_mean
        sigma   = torch.sqrt(bn2d_module.running_var + bn2d_module.eps)
        gamma   = bn2d_module.weight
        beta    = bn2d_module.bias
        eps_out = out_tunnel_module._eps_in
        assert torch.all(eps_out == qrelu_module.scale)

        broadcast_shape = tuple(1 if i != 1 else mi.numel() for i, _ in enumerate(range(0, len(shape))))
        mi    = mi.reshape(broadcast_shape)
        sigma = sigma.reshape(broadcast_shape)
        gamma = gamma.reshape(broadcast_shape)
        beta  = beta.reshape(broadcast_shape)

        gamma_int = torch.floor(self._D * (eps_in * gamma)             / (sigma * eps_out))
        beta_int  = torch.floor(self._D * (-mi * gamma + beta * sigma) / (sigma * eps_out))

        new_module = Requant(mul=gamma_int, add=beta_int, zero=qrelu_module.zero, n_levels=qrelu_module.n_levels, D=self._D)

        self._counter += 1
        new_target = '_'.join([self._name.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        query_in_tunnel  = ap.apcore[self._in_tunnel_node]
        query_bn2d       = ap.apcore[self._bn2d_node]
        query_qrelu      = ap.apcore[self._qrelu_node]
        query_out_tunnel = ap.apcore[self._out_tunnel_node]

        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(query_in_tunnel):
            new_node = g.graph.call_module(new_target, args=(query_in_tunnel,))
        query_out_tunnel.replace_input_with(query_qrelu, new_node)

        g.delete_submodule(query_qrelu.target)
        g.graph.erase_node(query_qrelu)
        g.delete_submodule(query_bn2d.target)
        g.graph.erase_node(query_bn2d)

        in_tunnel_module.set_eps_out(torch.ones_like(in_tunnel_module._eps_out))  # TODO: must have the same shape as the input eps
        out_tunnel_module.set_eps_in(torch.ones_like(out_tunnel_module._eps_in))   # TODO: must have the same shape as the output eps

        return g
