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

import torch.nn as nn
import torch.fx as fx
from typing import Tuple, Dict

from .applicationpoint import ActivationNode
from .activationspecification import ActivationSpecification
from quantlib.editing.editing.editors import Applier
from quantlib.editing.graphs.fx import FxNodeArgType


class ActivationReplacer(Applier):

    def __init__(self, specification: ActivationSpecification):

        if not isinstance(specification, ActivationSpecification):
            raise TypeError

        super(ActivationReplacer, self).__init__()
        self._specification = specification

    @property
    def specification(self) -> ActivationSpecification:
        return self._specification

    def from_nonmodular(self, n: fx.Node) -> Tuple[nn.Module, Tuple[FxNodeArgType, ...], Dict[str, FxNodeArgType]]:

        # unpack arguments to the input `fx.Node`...
        args, kwargs = n.args, n.kwargs
        # ...then split both positional arguments...
        call_args, instantiation_args = args[0:1], args[1:]
        # ...and keyword arguments into call (i.e., runtime) arguments and instantiation (i.e., creation) arguments
        call_kwargs = {k: v for k, v in kwargs.items() if k in ('input',)}
        instantiation_kwargs = {k: v for k, v in kwargs.items() if k not in ('input',)}

        if n in self.specification.targets.inplace:
            instantiation_kwargs['inplace'] = True

        module = self.specification.module_class(*instantiation_args, **instantiation_kwargs)

        return module, call_args, call_kwargs

    def _apply(self, g: fx.GraphModule, ap: ActivationNode, id_: str) -> fx.GraphModule:

        node = ap.node

        # compute the pieces of information to build the replacement
        new_target = id_
        module, call_args, call_kwargs = self.from_nonmodular(node)

        # perform the replacement
        g.add_submodule(new_target, module)   # add the replacement node, ...
        with g.graph.inserting_before(node):  # ...link it to the inbound part of the graph, ...
            new_node = g.graph.call_module(new_target, args=call_args, kwargs=call_kwargs)
        node.replace_all_uses_with(new_node)  # ...link it to the outbound part of the graph, ...
        g.graph.erase_node(node)              # ...then remove the old match

        return g
