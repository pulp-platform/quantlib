# 
# tracing.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

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
