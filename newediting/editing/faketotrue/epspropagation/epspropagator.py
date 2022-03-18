from collections import OrderedDict
import copy
import torch
import torch.fx as fx
from typing import Dict, Union

from .propagationrules import UNDEFINED_EPS, is_eps_annotated
from .propagationrules import _module_2_epspec
from .propagationrules import _method_2_epspec
from ...editors.editors import Annotator
from ....graphs import FXOPCODE_PLACEHOLDER, FXOPCODE_OUTPUT, FXOPCODE_CALL_MODULE, FXOPCODE_CALL_METHOD
from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.newutils import quantlib_err_header


class EpsPropagator(Annotator):
    """A class to annotate ``fx.GraphModule``s with quantiser scales."""

    def __init__(self):
        name = 'EpsPropagator'
        super(EpsPropagator, self).__init__(name)

    @staticmethod
    def clear_eps(g: fx.GraphModule):
        """Remove all scale annotations from the given graph."""
        for n in g.graph.nodes:
            try:
                del n.meta['eps']
            except KeyError:
                pass

    @staticmethod
    def returns_qtensor(n: fx.Node):
        return is_eps_annotated(n) and not torch.any(n.meta['eps'].isnan())

    def apply(self, g: fx.GraphModule, input_eps: Dict[str, Union[float, torch.Tensor]] = OrderedDict([])) -> fx.GraphModule:

        # clear input graph from old scale annotations
        EpsPropagator.clear_eps(g)

        # canonicalise input scales data type
        for name, eps in input_eps.items():
            if isinstance(eps, float):
                input_eps[name] = torch.Tensor([eps])

        # verify that there are no spurious scale specifications
        specified_eps_names = set(input_eps.keys())
        placeholder_names   = {n.name for n in g.graph.nodes if n.op in FXOPCODE_PLACEHOLDER}
        unknown_names       = specified_eps_names.difference(placeholder_names)
        if len(unknown_names) > 0:
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"was passed scale specifications for unknown placeholder nodes {unknown_names}.")

        for n in g.graph.nodes:
            # I assume that the nodes in the `fx.Graph` are topologically sorted.
            # Although the documentation of torch.fx is a bit opaque about this property,
            # we observe that if they were not topologically sorted then the IR might be
            # invalid. In fact, imagine that an `fx.Node` in position :math:`i` were
            # taking in input at least an `fx.Node` in position :math:`j > i`. This would
            # mean that the IR is suggesting that is possible to execute node :math:`i`
            # before node :math:`j`, violating the dependency constraint. I.e., an error.

            if n.op in FXOPCODE_PLACEHOLDER:  # annotate by looking-up into `input_eps` (either provided by the user or by the caller function)
                n.meta['eps'] = input_eps[n.name]
                # if n.name in input_eps.keys():
                #     n.meta['eps'] = input_eps[n.name]
                # else:
                #     n.meta['eps'] = UNDEFINED_EPS

            elif n.op in FXOPCODE_CALL_MODULE:
                m = g.get_submodule(n.target)
                try:
                    type_ = type(m) if not isinstance(m, _QModule) else _QModule
                    epspec = _module_2_epspec[type_]
                    epspec.function(n, m, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    # I assume that each `call_module` `fx.Node` yields a `torch.Tensor` which has a
                    # valid semantic with respect to the functionality that the network is designed
                    # to solve (e.g., a feature map). Therefore, if the epsilon propagation rule for
                    # a given `call_module` node is not defined, I return the "undefined" value ('NaN').
                    n.meta['eps'] = UNDEFINED_EPS

            elif n.op in FXOPCODE_CALL_METHOD:
                try:
                    epspec = _method_2_epspec[n.target]
                    epspec.function(n, None, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    # Differently from `call_module` `fx.Node`s, there are `call_method` nodes that
                    # yield values which do not have a valid semantic with respect to the functionality
                    # that the network is designed to achieve (e.g., evaluating the `size` of a given
                    # `torch.Tensor`). Therefore, the default behaviour here is skipping the annotation.
                    continue

            elif n.op in FXOPCODE_OUTPUT:  # ensure that it has exactly one argument, then just push its quantum forward
                assert len(n.args) == 1 and len(n.kwargs) == 0, "[QuantLab] Output nodes should read the output of a single operation."
                n.meta['eps'] = next(iter(n.args)).meta['eps']

            else:
                raise ValueError("[QuantLab] Epsilon-annotation of `fx.Node`s with opcode {} is not supported.".format(n.op))

        return g
