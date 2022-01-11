import itertools
import torch
import torch.nn as nn
import torch.fx as fx

from typing import Any, List, Dict, Callable, Union

import quantlib.algorithms


__all__ = [
    'CustomTracer',
    'QLTracer',
    'custom_symbolic_trace',
]


class CustomTracer(fx.Tracer):
    """An ``fx.Tracer`` with custom granularity.

    This class is the blueprint for ``fx.Tracer``s that interpret user-defined
    ``nn.Module`` classes as atomic ``fx.Node``s during symbolic tracing.
    """

    def __init__(self,
                 leaf_types: Union[None, List[torch.nn.Module]] = None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        if (leaf_types is None) or (len(leaf_types) == 0):
            raise UserWarning("[QuantLab] `CustomTracer`'s constructor received an empty list of leaf types; consider using a standard `fx.Tracer`.")
        self._leaf_types = () if leaf_types is None else tuple(leaf_types)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """Extend the base class check to custom ``nn.Module``s."""
        fxtracer_cond = super().is_leaf_module(m, module_qualified_name)
        custom_cond = isinstance(m, self._leaf_types)
        return fxtracer_cond or custom_cond


class QLTracer(CustomTracer):

    def __init__(self,
                 ql_algorithms: Union[str, List[str]],
                 *args,
                 **kwargs):
        """Create a ``CustomTracer`` for the target QL-supported algorithms.

        This function resolves strings (acronyms of QL-supported algorithms)
        into collections of QuantLab ``nn.Module``s, then creates and returns
        a ``CustomTracer`` which will trace such ``nn.Module``s as atoms.

        This function is part of QuantLib's public interface. Therefore, we
        want to provide a flexible way for users to specify the behaviour,
        although this requires casting inputs to a valid form before the core
        processing.
        """

        # canonicalise input
        if isinstance(ql_algorithms, str):
            ql_algorithms = [ql_algorithms]
        ql_algorithms = [a.upper() for a in ql_algorithms]  # TODO: this instruction assumes that we ALWAYS use upper-case acronyms for our supported algorithms

        # validity
        wrong_keys = set(ql_algorithms).difference(set(quantlib.algorithms.QLModules.keys()))
        if len(wrong_keys) > 0:
            raise ValueError("[QuantLab] QuantLab does not support the following algorithms: {}.".format(wrong_keys))

        # retrieve the ``nn.Module``s to be traced as atoms
        ql_modules = list(itertools.chain(*[quantlib.algorithms.QLModules[k] for k in ql_algorithms]))

        super().__init__(ql_modules, *args, **kwargs)


def custom_symbolic_trace(root: Union[Callable, nn.Module],
                          concrete_args: Union[None, Dict[str, Any]] = None,
                          enable_cpatching: bool = False,
                          tracer: Union[None, fx.Tracer] = None) -> fx.GraphModule:
    """Extend ``fx.symbolic_trace`` to tracings with custom granularity."""
    # resolve to PyTorch's "standard library"
    if tracer is None:
        gm = fx.symbolic_trace(root, concrete_args=concrete_args, enable_cpatching=enable_cpatching)

    # use ``CustomTracer``s to treat specific ``Module``s as atoms during symbolic tracing
    else:
        graph = tracer.trace(root, concrete_args)
        name = root.__class__.__name__ if isinstance(
            root, torch.nn.Module) else root.__name__
        gm = fx.GraphModule(tracer.root, graph, name)

    return gm
