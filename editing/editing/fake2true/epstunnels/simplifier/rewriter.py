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

from .finder import EpsTunnelConstructFinder
from .applier import EpsTunnelConstructApplier
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


class EpsTunnelConstructSimplifier(Rewriter):

    def __init__(self):
        super(EpsTunnelConstructSimplifier, self).__init__('EpsTunnelConstructSimplifier',
                                                           quantlib_symbolic_trace,
                                                           EpsTunnelConstructFinder(),
                                                           EpsTunnelConstructApplier())
