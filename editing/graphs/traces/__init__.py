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

import os
from collections import OrderedDict
import networkx as nx

from .trace import __TRACES_LIBRARY__
from .. import graphs

def load_traces_library(modules=None):

    mod_2_trace_dir = {}
    for root, dirs, files in os.walk(__TRACES_LIBRARY__):
        if len(dirs) == 0:  # terminal directories contain only trace files (graphviz, networkx)
            mod_2_trace_dir[os.path.basename(root)] = root

    if modules is None:
        modules = list(mod_2_trace_dir.keys())  # beware: there is no guarantee on the order in which the rescoping rules will be returned!

    libtraces = OrderedDict()
    for mod_name in modules:

        L = nx.read_gpickle(os.path.join(mod_2_trace_dir[mod_name], 'networkx'))
        VK = {n for n in L.nodes if L.nodes[n]['partition'] == graphs.Bipartite.CONTXT}
        for n in L.nodes:
            del L.nodes[n]['partition']
        K = L.subgraph(VK)

        libtraces[mod_name] = (L, K)

    return libtraces
