# 
# matching.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

from torch import fx
from torch.fx.subgraph_rewriter import Match

__all__ = ['gm_modules',
           'module_of_node',
           'named_module_nodes',
           'get_qualified_prefix',
           'modules_of_match']

def gm_modules(gm : fx.GraphModule):
    return dict(gm.named_modules())

def module_of_node(gm : fx.GraphModule, node : fx.Node):
    assert node.op == "call_module", "module_of_node can only be called on 'call_module' nodes!"

    return gm.get_submodule(node.target)

# convenience iterator to get all named modules and associated nodes
# yields (name, node, module) tuples
def named_module_nodes(gm : fx.GraphModule):
    for n in gm.graph.nodes:
        if n.op == "call_module":
            yield n.target, n, module_of_node(gm, n)

def get_qualified_prefix(target : str):
    spl = target.split('.')
    return '.'.join(spl[:-1])

def modules_of_match(gm : fx.GraphModule, m : Match):
    modules = gm_modules(gm)
    return [modules[m.target] for k, m in m.nodes_map.items() if k.op == 'call_module'][::-1]
