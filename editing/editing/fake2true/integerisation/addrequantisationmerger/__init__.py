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

from torch import fx
import torch.nn as nn
import quantlib.editing.graphs as qg
import quantlib.editing.editing as qe
import torch

from quantlib.editing.graphs.nn import Requantisation

class AddRequantisationPattern(nn.Module):

    def __init__(self):
        super(AddRequantisationPattern, self).__init__()
        self.requant1 = Requantisation(mul=torch.ones(1), add=torch.zeros(1), zero=torch.zeros(1), n_levels=torch.ones(1))
        self.requant2 = Requantisation(mul=torch.ones(1), add=torch.zeros(1), zero=torch.zeros(1), n_levels=torch.ones(1))
        self.requanto = Requantisation(mul=torch.ones(1), add=torch.zeros(1), zero=torch.zeros(1), n_levels=torch.ones(1))
    
    def forward(self, x, y):
        x = self.requant1(x)
        x = self.requant2(x)
        x = x + y
        x = self.requanto(x)
        return x

class AddRequantisationMergerApplier(qe.editors.nnmodules.NNModuleApplier):

    def __init__(self, pattern: qe.editors.nnmodules.GenericNNModulePattern):
        super(AddRequantisationMergerApplier, self).__init__(pattern)

    def _apply(self, g: fx.GraphModule, ap: qe.editors.nnmodules.NodesMap, id_: str) -> fx.GraphModule:

        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_requant1 = name_to_match_node['requant1']
        node_requant2 = name_to_match_node['requant2']

        try:
            name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        except RuntimeError:
            # this means this particular application point has already been managed
            return g
        module_requant1 = name_to_match_module['requant1']
        module_requant2 = name_to_match_module['requant2']

        # change zero, n_levels of first requant module
        clip_lo = max(module_requant1.zero, module_requant2.zero)
        clip_hi = min(module_requant1.zero + module_requant1.n_levels, module_requant2.zero + module_requant2.n_levels)
        n_levels = clip_hi - clip_lo
        zero = clip_lo
        print(module_requant1.zero, module_requant2.zero, zero)
        print(module_requant1.n_levels, module_requant2.n_levels, n_levels)

        # change mul, add, div of first requant module
        mul = torch.floor((module_requant1.mul * module_requant2.mul) / module_requant1.div)
        add = torch.floor((module_requant1.add * module_requant2.mul + module_requant1.div * module_requant2.add) / module_requant1.div)
        div = module_requant2.div
        print(module_requant1.mul, module_requant2.mul, mul)
        print(module_requant1.add, module_requant2.add, add)
        print(module_requant1.div, module_requant2.div, div)

        # create module
        new_module = Requantisation(mul=mul, add=add, zero=zero, n_levels=n_levels, D=div)

        # add the new module to graph
        new_target = id_
        g.add_submodule(new_target, new_module)
        new_input = next(iter(node_requant1.all_input_nodes))
        with g.graph.inserting_after(new_input):
            new_node = g.graph.call_module(new_target, args=(new_input,))
        node_requant2.replace_all_uses_with(new_node)
        node_requant1.replace_all_uses_with(new_node)

        # ...and delete the old operation
        g.delete_submodule(node_requant2.target)
        g.graph.erase_node(node_requant2)
        g.delete_submodule(node_requant1.target)
        g.graph.erase_node(node_requant1)
        g.graph.lint()

        return g

class AddRequantisationMerger(qe.editors.nnmodules.NNModuleRewriter):

    def __init__(self):
        # create pattern
        requantisation_withcheckers = qe.editors.nnmodules.NNModuleWithCheckers(AddRequantisationPattern(), {})
        requantisation_pattern = qe.editors.nnmodules.GenericNNModulePattern(qg.fx.quantlib_symbolic_trace, requantisation_withcheckers)
        # create matcher and applier
        finder = qe.editors.nnmodules.GenericGraphMatcher(requantisation_pattern)
        applier = AddRequantisationMergerApplier(requantisation_pattern)
        # link pattern, matcher, and applier into the rewriter
        super(AddRequantisationMerger, self).__init__('AddRequantisationMerger', requantisation_pattern, finder, applier)

__all__ = [
    'AddRequantisationMerger',
]
