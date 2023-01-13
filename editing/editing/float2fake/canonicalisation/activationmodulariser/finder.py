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

from .applicationpoint import ActivationNode
from .activationspecification import ActivationSpecification
from quantlib.editing.editing.editors import Finder
from quantlib.editing.graphs.fx import FXOpcodeClasses


class ActivationFinder(Finder):

    def __init__(self, specification: ActivationSpecification):

        if not isinstance(specification, ActivationSpecification):
            raise TypeError

        super(ActivationFinder, self).__init__()
        self._specification = specification

    @property
    def specification(self) -> ActivationSpecification:
        return self._specification

    def find(self, g: fx.GraphModule) -> List[ActivationNode]:
        nonmodular_nodes = filter(lambda n: (n.op in FXOpcodeClasses.CALL_NONMODULAR.value), g.graph.nodes)
        nonmodular_targets = filter(lambda n: (n.target in self.specification.targets), nonmodular_nodes)
        return [ActivationNode(n) for n in nonmodular_targets]

    def check_aps_commutativity(self, aps: List[ActivationNode]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))
