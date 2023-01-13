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

from .applicationpoint import ApplicationPoint


class Finder(object):

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        raise NotImplementedError

    def check_aps_commutativity(self, aps: List[ApplicationPoint]) -> bool:
        """Verify that the application points do not overlap.

        Passing this test ensures that the rewritings of the different
        application points can commute. Therefore, this avoids the need of
        recomputing the application points in-between applications of the same
        ``Rewriter``.

        """
        raise NotImplementedError
