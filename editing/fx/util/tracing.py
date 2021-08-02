from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, Type, List, Callable, Union

import torch
from torch import fx, nn

__all__ = ['LeafTracer',
           'custom_symbolic_trace']

class LeafTracer(fx.Tracer):
    # Allows tracing modules with custom granularity: Any modules of a type
    # contained in the leaf_types list will not be traced through and will
    # instead be represented as call_module nodes.
    def __init__(self, leaf_types: List[torch.nn.Module] = None, *args, **kwargs):
        self.leaf_types = [] if leaf_types is None else leaf_types
        super().__init__(*args, **kwargs)

    def is_leaf_module(self, m : torch.nn.Module, module_qualified_name : str):
        base_condition = super(LeafTracer, self).is_leaf_module(m, module_qualified_name)

        return base_condition or isinstance(m, tuple(self.leaf_types))


def custom_symbolic_trace(root : Union[Callable, nn.Module], concrete_args : Optional[Dict[str, Any]] = None, enable_cpatching: bool = False, tracer : Optional[fx.Tracer] = None):
    if tracer is None:
        tracer = fx.Tracer(enable_cpatching=enable_cpatching)

    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return fx.GraphModule(tracer.root, graph, name)
