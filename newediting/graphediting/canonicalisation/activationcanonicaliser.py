import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import List, Tuple, Dict, Any, Optional

from ..gebase import GraphRewriter, ApplicationPoint
from ...tracing import _OPCODES_NONMODULAR
from quantlib.newutils import quantlib_err_header


# first enumerate all the alternative descriptions of the canonical forms...
_MODULAR_TO_NONMODULAR = {
    nn.ReLU:      (torch.relu, torch.relu_, F.relu, F.relu_),
    nn.ReLU6:     (F.relu6,),
    nn.LeakyReLU: (F.leaky_relu, F.leaky_relu_),
}

# ...then compute the disambiguation mapping
_NONMODULAR_TO_MODULAR = {
    v: k for k, t in _MODULAR_TO_NONMODULAR.items() for v in t
}


_INPLACEABLE_CALLS = (
    *_MODULAR_TO_NONMODULAR[nn.ReLU],
    *_MODULAR_TO_NONMODULAR[nn.ReLU6],
    *_MODULAR_TO_NONMODULAR[nn.LeakyReLU],
)


class ActivationCanonicaliser(GraphRewriter):
    """Convert calls to activation functions to activation ``nn.Module``s."""

    def __init__(self):
        name = 'ActivationCanonicaliser'
        super(ActivationCanonicaliser, self).__init__(name)

    def find_application_points(self, data_gm: fx.GraphModule) -> List[fx.Node]:
        candidate_matches = list(filter(lambda n: n.op in _OPCODES_NONMODULAR, data_gm.graph.nodes))
        candidate_matches = list(filter(lambda n: n.target in _NONMODULAR_TO_MODULAR.keys(), candidate_matches))
        return candidate_matches

    @staticmethod
    def _convert_nonmodular_node(n: fx.Node) -> Tuple[nn.Module, Tuple[Any], Dict[str, Any]]:

        call_args, instantiation_args = n.args[0:1], n.args[1:]
        call_kwargs = {k: v for k, v in n.kwargs.items() if k in ('input',)}
        instantiation_kwargs = {k: v for k, v in n.kwargs.items() if k not in ('input',)}

        if n.target in _INPLACEABLE_CALLS:
            instantiation_kwargs['inplace'] = True

        module = _NONMODULAR_TO_MODULAR[n.target](*instantiation_args, **instantiation_kwargs)

        return module, call_args, call_kwargs

    def _apply(self, data_gm: fx.GraphModule, ap: fx.Node):

        if ap not in data_gm.graph.nodes:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "the application point must be a node of the data graph.")

        # compute the pieces of information to build the replacement
        module, call_args, call_kwargs = ActivationCanonicaliser._convert_nonmodular_node(ap)

        # compute the new target name
        self._counter += 1
        target = '_'.join([self._name.upper(), module.__class__.__name__.upper(), str(self._counter)])

        # perform the replacement
        data_gm.add_submodule(target, module)     # add the replacement node, ...
        with data_gm.graph.inserting_before(ap):  # ...link it to the inbound part of the graph, ...
            new_node = data_gm.graph.call_module(target, args=call_args, kwargs=call_kwargs)
        ap.replace_all_uses_with(new_node)        # ...link it to the outbound part of the graph, ...
        data_gm.graph.erase_node(ap)              # ...then remove the old match

        self._polish_graphmodule(data_gm)

    def apply(self, data_gm: fx.GraphModule, ap: Optional[ApplicationPoint] = None) -> fx.GraphModule:
        # if no specific application point is provided, select the first application point found by the `GraphWriter`'s automatic procedure
        if ap is None:
            ap = self._select_ap(data_gm)
        self._apply(data_gm, ap)
        return data_gm
