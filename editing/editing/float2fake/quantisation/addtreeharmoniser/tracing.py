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

from functools import partial

from quantlib.editing.graphs.fx import QuantLibTracer, custom_symbolic_trace
from quantlib.editing.graphs.nn import HarmonisedAdd


class QuantLibHarmonisedAddTracer(QuantLibTracer):
    def __init__(self):
        other_leaf_types = (HarmonisedAdd,)
        super(QuantLibHarmonisedAddTracer, self).__init__(other_leaf_types)


quantlib_harmonisedadd_symbolic_trace = partial(custom_symbolic_trace, tracer=QuantLibHarmonisedAddTracer())
