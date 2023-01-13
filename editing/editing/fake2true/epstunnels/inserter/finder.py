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
from typing import List

from .applicationpoint import EpsTunnelNode
from quantlib.editing.editing.editors import Finder
from quantlib.editing.editing.fake2true.annotation import EpsPropagator


class EpsTunnelInserterFinder(Finder):

    def find(self, g: fx.GraphModule) -> List[EpsTunnelNode]:
        aps = [EpsTunnelNode(n) for n in g.graph.nodes if EpsPropagator.returns_qtensor(n)]
        return aps

    def check_aps_commutativity(self, aps: List[EpsTunnelNode]) -> bool:
        return len(aps) == len(set(map(lambda ap: ap.node, aps)))
