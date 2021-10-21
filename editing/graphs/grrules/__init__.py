# 
# __init__.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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

from collections import OrderedDict

from .seeker import Seeker
from .dporules import *
from .hlprrules import *
from .. import traces


def load_rescoping_rules(modules=None):

    libtraces = traces.load_traces_library(modules=modules)

    librules = OrderedDict()
    for mod_name, (L, K) in libtraces.items():
        if mod_name == 'ViewFlattenNd':
            librules[mod_name] = ManualRescopingRule(L, K, 'torch.view')  # TODO: mind quantlib.editing.graphs/graphs/graphs.py:L205
        else:
            librules[mod_name] = AutoRescopingRule(L, K)

    return librules

