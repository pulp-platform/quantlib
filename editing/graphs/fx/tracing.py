# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
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

from functools import partial
import torch.nn as nn
import torch.fx as fx
from typing import Tuple, Dict, Any, Union, Optional, Callable, Type

from ..nn import EpsTunnel, Requantisation
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.utils import quantlib_err_header


class CustomTracer(fx.Tracer):
    """An ``fx.Tracer`` with custom granularity.

    This class is the blueprint for ``fx.Tracer``s that interpret user-defined
    ``nn.Module`` classes as atomic ``fx.Node``s during symbolic tracing.
    """

    def __init__(self,
                 leaf_types: Tuple[Type[nn.Module], ...],
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        if len(leaf_types) == 0:
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "requires a non-empty list of `nn.Module`s to be treated as leaves.")
        self._leaf_types = leaf_types

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """Extend the base class check to custom ``nn.Module``s."""
        fxtracer_cond = super().is_leaf_module(m, module_qualified_name)
        custom_cond   = isinstance(m, self._leaf_types)
        return fxtracer_cond or custom_cond


class QuantLibTracer(CustomTracer):

    def __init__(self, other_leaf_types: Tuple[Type[nn.Module], ...] = tuple(), *args, **kwargs):
        """An ``fx.Tracer`` treating QuantLib ``nn.Module``s as leaves.

        Users can sub-class it by passing additional ``nn.Module`` classes
        that should be treated as leaves. This functionality can be useful,
        for instance, when creating containers of ``_QModule``s.

        """
        quantlib_leaf_types = (_QModule, EpsTunnel, Requantisation,)
        leaf_types = (*quantlib_leaf_types, *other_leaf_types)
        super().__init__(leaf_types=leaf_types, *args, **kwargs)


SymbolicTraceFnType = Callable[[fx.Tracer, Union[Callable, nn.Module], Optional[Dict[str, Any]]], fx.GraphModule]


def custom_symbolic_trace(tracer: CustomTracer,
                          root: Union[Callable, nn.Module],
                          concrete_args: Optional[Dict[str, Any]] = None) -> fx.GraphModule:

    graph = tracer.trace(root, concrete_args)
    name  = root.__class__.__name__ if isinstance(root, nn.Module) else root.__name__
    gm    = fx.GraphModule(tracer.root, graph, name)

    return gm


quantlib_symbolic_trace = partial(custom_symbolic_trace, tracer=QuantLibTracer())
