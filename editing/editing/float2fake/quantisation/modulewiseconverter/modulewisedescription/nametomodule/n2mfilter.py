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

"""This module implements a system of functions to filter ``NameToModule``s.

The system is closed with respect to comopsition, since each filter takes in
input a ``NameToModule`` object and yields as output another ``NameToModule``.

"""

import functools
import re
import torch.nn as nn
from typing import Tuple, Callable, Type

from .nametomodule import NameToModule


N2MFilter = Callable[[NameToModule], NameToModule]


def null_filter(name_to_module: NameToModule) -> NameToModule:
    result = name_to_module.copy()
    return result


def regex_filter(name_to_module: NameToModule, regex: str) -> NameToModule:
    result = NameToModule([(n, m) for n, m in name_to_module.items() if re.match(regex, n)])
    return result


def types_filter(name_to_module: NameToModule, types: Tuple[Type[nn.Module], ...]) -> NameToModule:
    result = NameToModule([(n, m) for n, m in name_to_module.items() if isinstance(m, types)])
    return result


def inclusive_names_filter(name_to_module: NameToModule, names: Tuple[str, ...]) -> NameToModule:
    result = NameToModule([(n, m) for n, m in name_to_module.items() if n in names])
    return result


def exclusive_names_filter(name_to_module: NameToModule, names: Tuple[str, ...]) -> NameToModule:
    result = NameToModule([(n, m) for n, m in name_to_module.items() if n not in names])
    return result


def compose(*filters: N2MFilter) -> N2MFilter:
    return functools.reduce(lambda f, g: lambda n2m: g(f(n2m)), filters)  # https://www.youtube.com/watch?v=ka70COItN40&t=1346s
