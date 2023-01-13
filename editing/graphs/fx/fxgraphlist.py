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

from __future__ import annotations

import torch.nn as nn
import torch.fx as fx
from typing import NamedTuple, Optional

from .fxnodes import FXOpcodeClasses


class FXNodePayload(NamedTuple):
    op:     str  # `fx` opcode
    module: Optional[nn.Module] = None


class LightFXNode(NamedTuple):
    id_:         int
    node:        fx.Node
    gm:          fx.GraphModule

    def __str__(self) -> str:
        return f"({self.id_}, {self.node}, {self.payload.module})"

    def __repr__(self):
        return self.__str__()

    @property
    def payload(self) -> FXNodePayload:
        op_ = self.node.op
        module_ = self.gm.get_submodule(target=self.node.target) if (op_ in FXOpcodeClasses.CALL_MODULE.value) else None
        return FXNodePayload(op=op_, module=module_)

    @property
    def upstream(self) -> FXGraphList:
        return FXGraphList(LightFXNode(id_=i, node=u, gm=self.gm) for (i, u) in enumerate(self.gm.graph.nodes) if (u in self.node.all_input_nodes))

    @property
    def downstream(self) -> FXGraphList:
        return FXGraphList(LightFXNode(id_=i, node=d, gm=self.gm) for (i, d) in enumerate(self.gm.graph.nodes) if (d in self.node.users))


class FXGraphList(list):
    """A list sub-class accepting only ``LightFXNode`` items."""

    def append(self, item):
        if not isinstance(item, LightFXNode):
            raise TypeError
        super(FXGraphList, self).append(item)

    def insert(self, index, item):
        if not isinstance(item, LightFXNode):
            raise TypeError
        super(FXGraphList, self).insert(index, item)

    def __add__(self, item):
        if not isinstance(item, LightFXNode):
            raise TypeError
        super(FXGraphList, self).__add__(item)

    def __iadd__(self, item):
        if not isinstance(item, LightFXNode):
            raise TypeError
        super(FXGraphList, self).__iadd__(item)

    def __str__(self) -> str:
        """When printed, this string shows a nicely formatted list of
        ``LightFXNode``s.

        This functionality is useful, for instance, when users want to
        visually inspect an ``fx.GraphModule`` to debug ``Editor``s.

        This method overwrites the default behaviour of Python lists'
        ``__str__`` dunder method, which resolves to calling the ``__repr__``
        method of each item in the list (https://stackoverflow.com/a/727779).

        """

        # overwrite the default behaviour of Python lists' `__str__` method
        max_id_chars = max(map(lambda node: len(str(node.id_)), self))
        max_name_chars = max(map(lambda node: len(node.node.name), self))

        str_ = f""
        for node in self:
            str_ += f"\n"
            str_ += "(" + str(node.id_).rjust(max_id_chars) + ", " + node.node.name.ljust(max_name_chars) + ", " + str(node.payload.module) + ")"

        return str_

    def show(self) -> None:
        print(self.__str__())
    

def fxgm_to_fxgl(gm: fx.GraphModule) -> FXGraphList:
    """Map an ``fx.GraphModule`` object to the corresponding ``FXGraphList``."""
    return FXGraphList(LightFXNode(id_=i, node=n, gm=gm) for (i, n) in enumerate(gm.graph.nodes))
