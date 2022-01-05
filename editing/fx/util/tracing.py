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
import itertools

import torch
from torch import fx, nn

from quantlib.algorithms.ana.ops import ANAModule

__all__ = [
    'TracerFactory',
    'CustomTracer',
    'custom_symbolic_trace',
]


class CustomTracer(fx.Tracer):
    """An ``fx.Tracer`` with custom granularity.

    This class is the blueprint for ``Tracer``s that interpret user-defined
    ``nn.Module`` classes as atomic ``fx.Node``s during symbolic tracing.
    """

    def __init__(self,
                 leaf_types: Optional[List[torch.nn.Module]] = None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self._leaf_types = () if leaf_types is None else tuple(leaf_types)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """Extend the base class check to custom ``nn.Module``s."""
        fxtracer_cond = super().is_leaf_module(m, module_qualified_name)
        custom_cond = isinstance(m, self._leaf_types)
        return fxtracer_cond or custom_cond


_QL_modules = {
    'ANA': [ANAModule],
}


class TracerFactory(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_tracer(ql_algorithms: Union[str, List[str]],
                   *args,
                   **kwargs) -> CustomTracer:
        """Create a ``CustomTracer`` for the target QAT algorithms.

        This function resolves strings (acronyms of QAT algorithms) into
        collections of QuantLab ``nn.Module``s, then creates and returns a
        ``CustomTracer`` which will trace such ``Module``s as atoms.

        This function is part of ``quantlib``'s public interface. Therefore,
        we want to provide a flexible way for users to specify the behaviour,
        although this requires casting inputs to a valid form before the core
        processing.
        """

        # canonicalise input
        if isinstance(ql_algorithms, (str, list)):
            if isinstance(ql_algorithms, str):
                ql_algorithms = [ql_algorithms]
            ql_algorithms = [a.upper() for a in ql_algorithms]

        else:
            raise TypeError("TracerFactory expects str or List[str], but type {} was passed.".format(type(ql_algorithms)))

        # validity
        wrong_keys = set(ql_algorithms).difference(set(_QL_modules.keys()))
        if len(wrong_keys) > 0:
            raise ValueError("QuantLab does not support algorithms {}.".format(wrong_keys))

        # retrieve the ``nn.Module``s to be traced as atoms
        ql_modules = list(itertools.chain(*[_QL_modules[k] for k in ql_algorithms]))
        return CustomTracer(ql_modules, *args, **kwargs)


def custom_symbolic_trace(root: Union[Callable, nn.Module],
                          concrete_args: Optional[Dict[str, Any]] = None,
                          enable_cpatching: bool = False,
                          tracer: Optional[fx.Tracer] = None) -> fx.GraphModule:
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
