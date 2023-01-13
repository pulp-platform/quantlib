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

from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern, PathGraphMatcher


class LinearOpIntegeriserMatcher(PathGraphMatcher):

    def __init__(self, pattern: NNSequentialPattern):
        super(LinearOpIntegeriserMatcher, self).__init__(pattern)
        # Despite the fact that `eps_out` is in the "body" of the linear
        # pattern, we allow for its matched `fx.Node` to have multiple users
        # since its output is not meant to change after the rewriting.
        pattern.set_leakable_nodes(pattern.name_to_pattern_node()['eps_out'])
