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

from .finder import FinalEpsTunnelRemoverFinder
from .applier import FinalEpsTunnelRemoverApplier
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace

# removes the output node eps-tunnel
class FinalEpsTunnelRemover(Rewriter):

    def __init__(self):
        super(FinalEpsTunnelRemover, self).__init__('FinalEpsTunnelRemover',
                                               quantlib_symbolic_trace,
                                               FinalEpsTunnelRemoverFinder(),
                                               FinalEpsTunnelRemoverApplier())
