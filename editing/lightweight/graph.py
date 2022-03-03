# 
# graph.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

from .node import LightweightNode

from typing import List


class LightweightGraph(object):

    def __init__(self, net: torch.nn.Module):

        self._net        = net
        self._nodes_list = LightweightGraph.build_nodes_list(self._net, nodes_list=[])

    @property
    def net(self) -> torch.nn.Module:
        return self._net

    @property
    def nodes_list(self) -> List[LightweightNode]:
        return self._nodes_list

    @staticmethod
    def build_nodes_list(parent_module: torch.nn.Module, parent_name: str = '', nodes_list: List[LightweightNode] = []):

        for name, child in parent_module.named_children():
            if len(list(child.children())) == 0:
                nodes_list.append(LightweightNode(name=parent_name + name, module=child))
            else:
                LightweightGraph.build_nodes_list(child, parent_name=parent_name + name + '.', nodes_list=nodes_list)

        return nodes_list

    def rebuild_nodes_list(self):
        self._nodes_list = LightweightGraph.build_nodes_list(self._net, nodes_list=[])

    def show_nodes_list(self):

        for lwn in self._nodes_list:
            print("{:30s} {}".format(lwn.name, lwn.type_))
