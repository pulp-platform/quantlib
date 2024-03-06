# 
# filters.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

import re

from ..node import LightweightNode
from typing import List


__all__ = [
    'NotFilter',
    'OrFilter',
    'AndFilter',
    'VariadicOrFilter',
    'VariadicAndFilter',
    'NameFilter',
    'TypeFilter',
    'SubTypeFilter'
]


class Filter(object):

    def __init__(self, *args):
        pass

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        raise NotImplementedError

    def __call__(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return self.find(nodes_list)

    def __neg__(self):
        return NotFilter(self)

    def __invert__(self):
        return NotFilter(self)

    def __and__(self, other):
        return AndFilter(self, other)

    def __or__(self, other):
        return OrFilter(self, other)


class NotFilter(Filter):

    def __init__(self, filter_: Filter):
        super(NotFilter, self).__init__()
        self._filter = filter_

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return [n for n in nodes_list if n not in self._filter(nodes_list)]

    def __repr__(self):
        return "".join(["(-", repr(self._filter), ")"])


class OrFilter(Filter):

    def __init__(self, filter_a: Filter, filter_b: Filter):
        super(OrFilter, self).__init__()
        self._filter_a = filter_a
        self._filter_b = filter_b

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:

        filter_a_nodes = self._filter_a(nodes_list)
        filter_b_nodes = self._filter_b(nodes_list)
        return filter_a_nodes + [n for n in filter_b_nodes if n not in filter_a_nodes]  # remove duplicates

    def __repr__(self):
        return "".join(["(", repr(self._filter_a), " | ", repr(self._filter_b), ")"])


class AndFilter(Filter):

    def __init__(self, filter_a: Filter, filter_b: Filter):
        super(AndFilter, self).__init__()
        self._filter_a = filter_a
        self._filter_b = filter_b

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:

        filter_a_nodes = self._filter_a(nodes_list)
        filter_b_nodes = self._filter_b(filter_a_nodes)

        return filter_b_nodes

    def __repr__(self):
        return "".join(["(", repr(self._filter_a), " & ", repr(self._filter_b), ")"])


class VariadicOrFilter(Filter):

    def __init__(self, *filters: Filter):
        super(VariadicOrFilter, self).__init__()
        self._filters = filters

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        filtered_nodes = []
        for f in self._filters:
            filtered_nodes += [n for n in f(nodes_list) if n not in filtered_nodes]

        return filtered_nodes

    def __repr__(self):
        return "".join(["("] + [repr(f) + " | " for f in self._filters[:-1]] + [repr(self._filters[-1]), ")"])


class VariadicAndFilter(Filter):

    def __init__(self, *filters):
        assert len(filters) >= 2
        super(VariadicAndFilter, self).__init__()
        self._filters = filters

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        filtered_nodes = nodes_list
        for f in self._filters:
            filtered_nodes = f(filtered_nodes)
        return filtered_nodes

    def __repr__(self):
        return "".join(["("] + [repr(f) + " & " for f in self._filters[:-1]] + [repr(self._filters[-1]), ")"])


class NameFilter(Filter):

    def __init__(self, regex: str):
        super(NameFilter, self).__init__()
        self._regex   = regex
        self._pattern = re.compile(self._regex)

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return list(filter(lambda n: self._pattern.match(n.name), nodes_list))

    def __repr__(self):
        return "".join([self.__class__.__name__, "('", self._regex, "')"])


class TypeFilter(Filter):

    def __init__(self, type_: type):
        super(TypeFilter, self).__init__()
        self._type = type_

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return list(filter(lambda n: n.type_ == self._type, nodes_list))

    @property
    def _type_str(self):
        return str(self._type).replace("<class '", "").replace("'>", "")

    def __repr__(self):
        return "".join([self.__class__.__name__, "(", self._type_str, ")"])

class SubTypeFilter(Filter):

    def __init__(self, type_: type):
        super(SubTypeFilter, self).__init__()
        self._type = type_

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return list(filter(lambda n: isinstance(n.module, self._type), nodes_list))

    @property
    def _type_str(self):
        return str(self._type).replace("<class '", "").replace("'>", "")

    def __repr__(self):
        return "".join([self.__class__.__name__, "(", self._type_str, ")"])

