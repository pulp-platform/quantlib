# 
# Author(s):
# Francesco Conti <f.conti@unibo.it>
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

import torch
import torch.fx as fx
from typing import List

from ..remover.applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Finder
from quantlib.editing.graphs.fx import FXOpcodeClasses
from quantlib.editing.graphs.nn import EpsTunnel


class FinalEpsTunnelRemoverFinder(Finder):

    def find(self, g: fx.GraphModule) -> List[EpsTunnelNode]:

        # find `EpsTunnel` `fx.Node`s
        module_nodes = filter(lambda n: (n.op in FXOpcodeClasses.CALL_MODULE.value), g.graph.nodes)

        # select only output nodes
        epstunnels = list(filter(lambda n: isinstance(g.get_submodule(target=n.target), EpsTunnel), module_nodes))  # since we consume the `filter` generator twice in the next lines, we must ensure that it does not get empty after the first consumption
        singleusertunnels = list(filter(lambda n: len(n.users) == 1, epstunnels))
        outputtunnels = list(filter(lambda n: list(n.users.keys())[0].op == "output", singleusertunnels))

        return [EpsTunnelNode(n) for n in list(outputtunnels)]

    def check_aps_commutativity(self, aps: List[EpsTunnelNode]) -> bool:
        return len(aps) == len(set(map(lambda ap: ap.node, aps)))
