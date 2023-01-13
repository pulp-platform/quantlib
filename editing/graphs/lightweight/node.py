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
from typing import NamedTuple, List


class LightweightNode(NamedTuple):
    """Attach a symbolic name to an ``nn.Module``."""

    name:   str
    module: nn.Module

    @property
    def qualified_path(self) -> List[str]:
        return self.name.split('.')

    def __eq__(self, other: LightweightNode) -> bool:
        cond_type = isinstance(other, LightweightNode) 
        cond_name = self.name == other.name
        cond_obj  = self.module is other.module
        return cond_type and cond_name and cond_obj


class LightweightNodeList(list):  # https://stackoverflow.com/a/24160909
    """A list sub-class accepting only ``LightweightNode`` items."""

    def append(self, item):
        if not isinstance(item, LightweightNode):
            raise TypeError
        super(LightweightNodeList, self).append(item)

    def insert(self, index, item):
        if not isinstance(item, LightweightNode):
            raise TypeError
        super(LightweightNodeList, self).insert(index, item)

    def __add__(self, item):
        if not isinstance(item, LightweightNode):
            raise TypeError
        super(LightweightNodeList, self).__add__(item)

    def __iadd__(self, item):
        if not isinstance(item, LightweightNode):
            raise TypeError
        super(LightweightNodeList, self).__iadd__(item)

    def __str__(self) -> str:
        """When printed, this string shows a nicely formatted list of named
        ``nn.Module``s.

        This functionality is useful, for instance, when users want to
        visually inspect a floating-point ``nn.Module`` to decide how to
        quantise it.

        This method overwrites the default behaviour of Python lists'
        ``__str__`` dunder method, which resolves to calling the ``__repr__``
        method of each item in the list (https://stackoverflow.com/a/727779).

        """

        # overwrite the default behaviour of Python lists' `__str__` method
        max_chars = max(map(lambda node: len(node.name), self))

        str_ = f""
        for node in self:
            str_ += f"\n"
            name = ".".join(reversed(node.qualified_path)).rjust(max_chars)
            str_ += f"{name}\t{node.module}"

        return str_

    def show(self) -> None:
        print(self.__str__())
