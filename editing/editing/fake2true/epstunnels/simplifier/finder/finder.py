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

from ..applicationpoint import CandidateEpsTunnelConstruct
from .algorithm import find_candidate_constructs, verify_candidate_construct
from quantlib.editing.editing.editors import Finder


class EpsTunnelConstructFinder(Finder):

    def find(self, g: fx.GraphModule) -> List[CandidateEpsTunnelConstruct]:
        candidate_constructs = find_candidate_constructs(g)
        constructs = list(filter(lambda cc: verify_candidate_construct(cc, g), candidate_constructs))
        return constructs

    def check_aps_commutativity(self, aps: List[CandidateEpsTunnelConstruct]) -> bool:
        return True  # TODO: implement the check!
