import torch
import torch.fx as fx
import copy
from typing import Dict, Union


from .customprop import _undefined_eps
from .customprop import _module_2_epspec
from .customprop import _method_2_epspec


# TODO: Should we define a base `Annotator` class?
#       Such a class would complement the `Rule/Pass` class, defining a
#       taxonomy of our objects in non-transformative ones (`Annotator`s) and
#       transformative ones (`Rule/Pass`es).

class EpsAnnotator(object):
    """An object to annotate an `fx.GraphModule` with quanta information."""

    def __init__(self, gm: fx.GraphModule):
        self._gm = gm

    @property
    def gm(self) -> fx.GraphModule:
        return self._gm

    @staticmethod
    def is_eps_annotated(n: fx.Node) -> bool:
        return 'eps' in n.meta.keys()

    @staticmethod
    def returns_qtensor(n: fx.Node):
        return EpsAnnotator.is_eps_annotated(n) and not n.meta['eps'].isnan()

    def clear(self):
        """Remove all epsilon-annotations from the graph."""
        for n in self.gm.graph.nodes:
            try:
                del n.meta['eps']
            except KeyError:
                pass

    def apply(self, inputs_eps: Dict[str, Union[float, torch.Tensor]] = {}) -> None:

        # canonicalise input
        for k, v in inputs_eps.items():
            if isinstance(v, float):
                inputs_eps[k] = torch.Tensor([v])

        for n in self.gm.graph.nodes:  # switch construct based on the `fx.Node`'s opcode
            # I assume that the nodes in the `fx.Graph` are topologically sorted.
            # Although the documentation of torch.fx is a bit opaque about this property,
            # we observe that if they were not topologically sorted then the IR might be
            # invalid. In fact, imagine that an `fx.Node` in position :math:`i` were
            # taking in input at least an `fx.Node` in position :math:`j > i`. This would
            # mean that the IR is suggesting that is possible to execute node :math:`i`
            # before node :math:`j`, violating the dependency constraint. I.e., an error.

            if EpsAnnotator.is_eps_annotated(n):
                raise RuntimeError("[QuantLab] `fx.Node` {} is already epsilon-annotated: you must clear the annotations before epsilon-annotating a graph.".format(n))

            if n.op == 'placeholder':  # annotate by looking-up into `input_epss` (either provided by the user or by the caller function)
                if n.name in inputs_eps.keys():
                    eps_out = inputs_eps[n.name]
                else:
                    eps_out = _undefined_eps

            elif n.op == 'call_module':
                m = self.gm.get_submodule(n.target)
                try:
                    epspec = _module_2_epspec[type(m)]
                    eps_out = epspec.function(n, m, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    # I assume that each `call_module` `fx.Node` yields a `torch.Tensor` which has a
                    # valid semantic with respect to the functionality that the network is designed
                    # to solve (e.g., a feature map). Therefore, if the epsilon propagation rule for
                    # a given `call_module` node is not defined, I return the "undefined" value ('NaN').
                    eps_out = _undefined_eps

            elif n.op == 'call_method':
                try:
                    epspec = _method_2_epspec[n.target]
                    eps_out = epspec.function(n, None, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    # Differently from `call_module` `fx.Node`s, there are `call_method` nodes that
                    # yield values which do not have a valid semantic with respect to the functionality
                    # that the network is designed to achieve (e.g., evaluating the `size` of a given
                    # `torch.Tensor`). Therefore, the default behaviour here is skipping the annotation.
                    continue

            elif n.op == 'output':  # ensure that it has exactly one argument, then just push its quantum forward
                assert len(n.args) == 1 and len(n.kwargs) == 0, "[QuantLab] Output nodes should read the output of a single operation."
                eps_out = next(iter(n.args)).meta['eps']

            else:
                raise ValueError("[QuantLab] Epsilon-annotation of `fx.Node`s with opcode {} is not supported.".format(n.op))

            n.meta['eps'] = eps_out
