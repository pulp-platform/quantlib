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

import itertools
import torch.fx as fx

from .applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Applier


class EpsTunnelRemoverApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: EpsTunnelNode, id_: str) -> fx.GraphModule:

        node = ap.node

        # the `fx.Node` is functionally equivalent to the identity, so we connect its (unique) input to all the outputs
        predecessors = {p for p in node.all_input_nodes}  # upstream
        assert len(predecessors) == 1
        successors = {s for s in node.users}  # downstream
        for p, s in itertools.product(predecessors, successors):
            s.replace_input_with(node, p)

        g.delete_submodule(node.target)
        g.graph.erase_node(node)

        return g
