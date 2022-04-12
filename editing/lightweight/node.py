# 
# node.py
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

import torch

from typing import List


class LightweightNode(object):

    def __init__(self, name: str, module: torch.nn.Module):
        self.name   = name
        self.module = module

    @property
    def path(self) -> List[str]:
        return self.name.split('.')

    @property
    def type_(self) -> type:
        return type(self.module)

    def __repr__(self) -> str:
        return f"LightweightNode(name={self.name}, module={self.module})"

    def __eq__(self, other) -> bool:
        # two nodes are equal if they have the same name and point to the same
        # nn.Module instance.
        if isinstance(other, LightweightNode):
            return (self.name == other.name) and (self.module is other.module)
        else:
            raise NotImplementedError
