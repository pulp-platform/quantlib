import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import List, Tuple, Dict

from quantlib.editing.editing.editors.editors import ApplicationPoint, Rewriter
from quantlib.editing.graphs import FXOPCODES_CALL_NONMODULAR
from quantlib.editing.graphs import FxNodeArgType, unpack_fxnode_arguments
from quantlib.utils import quantlib_err_header


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


class ActivationModulariser(Rewriter):
    """Convert calls to activation functions to activation ``nn.Module``s."""

    def __init__(self):
        name = 'ActivationModulariser'
        super(ActivationModulariser, self).__init__(name)

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:

        candidate_matches = list(filter(lambda n: n.op     in FXOPCODES_CALL_NONMODULAR,     g.graph.nodes))
        candidate_matches = list(filter(lambda n: n.target in _NONMODULAR_TO_MODULAR.keys(), candidate_matches))

        aps = [ApplicationPoint(rewriter=self, graph=g, apcore=n) for n in candidate_matches]

        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:

        # Were the application points computed by this `Rewriter`, and on the target `fx.GraphModule`?
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

        # Are the application points independent of each other? I.e., do rewritings commute?
        apcores = [ap.apcore for ap in aps]
        if len(apcores) != len(set(apcores)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "was passed duplicate application points. Ensure that all the application points have disjoint support.")

    @staticmethod
    def from_nonmodular(n: fx.Node) -> Tuple[nn.Module, Tuple[FxNodeArgType, ...], Dict[str, FxNodeArgType]]:

        args, kwargs = unpack_fxnode_arguments(n)

        call_args, instantiation_args = args[0:1], args[1:]
        call_kwargs = {k: v for k, v in kwargs.items() if k in ('input',)}
        instantiation_kwargs = {k: v for k, v in kwargs.items() if k not in ('input',)}

        if n.target in _INPLACEABLE_CALLS:
            instantiation_kwargs['inplace'] = True

        module = _NONMODULAR_TO_MODULAR[n.target](*instantiation_args, **instantiation_kwargs)

        return module, call_args, call_kwargs

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        # compute the pieces of information to build the replacement
        module, call_args, call_kwargs = ActivationModulariser.from_nonmodular(ap.apcore)

        # compute the new target name
        self._counter += 1
        target = '_'.join([self._name.upper(), module.__class__.__name__.upper(), str(self._counter)])

        # perform the replacement
        g.add_submodule(target, module)            # add the replacement node, ...
        with g.graph.inserting_before(ap.apcore):  # ...link it to the inbound part of the graph, ...
            new_node = g.graph.call_module(target, args=call_args, kwargs=call_kwargs)
        ap.apcore.replace_all_uses_with(new_node)  # ...link it to the outbound part of the graph, ...
        g.graph.erase_node(ap.apcore)              # ...then remove the old match

        self._polish_fxgraphmodule(g)

        return g
