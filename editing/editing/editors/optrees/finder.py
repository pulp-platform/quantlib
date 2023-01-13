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

import torch.fx as fx
from typing import Tuple, List, Set

from .applicationpoint import OpTree, OpSpec
from ..base import Finder


class OpTreeFinder(Finder):

    def __init__(self, opspec: OpSpec):
        super(OpTreeFinder, self).__init__()
        self._opspec = opspec

    @property
    def opspec(self) -> OpSpec:
        return self._opspec

    @staticmethod
    def _find_optrees(visited_nodes: Set[fx.Node],
                      opspec:        OpSpec,
                      dn:            fx.Node) -> Tuple[List[OpTree], List[OpTree]]:

        visited_nodes.add(dn)  # mark the node as visited

        current_optree:   List[OpTree] = []
        upstream_optrees: List[OpTree] = []

        if opspec.matches_opspec(dn):

            optree = OpTree(root=dn)

            for next_dn in dn.all_input_nodes:
                if next_dn in visited_nodes:
                    pass
                else:
                    child_optree, child_upstream_optrees = OpTreeFinder._find_optrees(visited_nodes, opspec, next_dn)
                    optree.merge(child_optree)
                    upstream_optrees.extend(child_upstream_optrees)

            if len(dn.users) <= 1:
                current_optree.append(optree)
            else:  # `1 < len(dn.users)`: `dn` is a branching point, therefore it must be the `OpTree` root
                upstream_optrees.append(optree)  # the parent `fx.Node` won't merge this node's `OpTree` into its own `OpTree`

        else:  # `not opspec.matches_opspec(dn)`

            for next_dn in dn.all_input_nodes:
                if next_dn in visited_nodes:
                    pass
                else:
                    child_optree, child_upstream_optrees = OpTreeFinder._find_optrees(visited_nodes, opspec, next_dn)
                    upstream_optrees.extend(child_upstream_optrees)
                    upstream_optrees.extend(child_optree)

        if not (0 <= len(current_optree) <= 1):
            raise RuntimeError

        return current_optree, upstream_optrees

    def find(self, data_gm: fx.GraphModule) -> List[OpTree]:

        optrees: List[OpTree] = []

        visited_nodes = set()
        for dn in reversed(data_gm.graph.nodes):

            if dn in visited_nodes:
                pass

            else:
                child_optree, child_upstream_optrees = OpTreeFinder._find_optrees(visited_nodes, self.opspec, dn)
                optrees.extend(child_upstream_optrees)
                optrees.extend(child_optree)

        return optrees

    def check_aps_commutativity(self, aps: List[OpTree]) -> bool:
        return True  # TODO: implement the check!
