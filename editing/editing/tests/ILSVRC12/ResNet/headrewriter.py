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

import torch
import torch.nn as nn
import torch.fx as fx

import quantlib.editing.graphs as qg
import quantlib.editing.editing as qe


class RNHead(nn.Module):

    def __init__(self):
        super(RNHead, self).__init__()
        self.eps = qg.nn.EpsTunnel(torch.Tensor([1.0]))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eps(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x


class RNHeadApplier(qe.editors.nnmodules.NNModuleApplier):

    def __init__(self, rn18headpattern: qe.editors.nnmodules.GenericNNModulePattern):
        super(RNHeadApplier, self).__init__(rn18headpattern)

    def _apply(self, g: fx.GraphModule, ap: qe.editors.nnmodules.NodesMap, id_: str) -> fx.GraphModule:

        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_lin = name_to_match_node['lin']

        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps = name_to_match_module['eps']
        module_lin = name_to_match_module['lin']

        assert module_eps.eps_out.numel() == 1
        assert len(node_lin.all_input_nodes) == 1

        # create the new module
        new_target = id_
        new_module = nn.Linear(in_features=module_lin.in_features, out_features=module_lin.out_features, bias=module_lin.bias is not None)
        new_weight = module_lin.weight.data.detach().clone() * module_eps.eps_out
        new_module.weight.data = new_weight
        if module_lin.bias is not None:
            new_bias = module_lin.bias.data.detach().clone()
            new_module.bias.data = new_bias

        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        linear_input = next(iter(node_lin.all_input_nodes))
        with g.graph.inserting_after(linear_input):
            new_node = g.graph.call_module(new_target, args=(linear_input,))
        node_lin.replace_all_uses_with(new_node)

        module_eps.set_eps_out(torch.ones_like(module_eps.eps_out))

        # ...and delete the old operation
        g.delete_submodule(node_lin.target)
        g.graph.erase_node(node_lin)

        return g


class RNHeadRewriter(qe.editors.nnmodules.NNModuleRewriter):

    def __init__(self):
        # create pattern
        rnheadwithcheckers = qe.editors.nnmodules.NNModuleWithCheckers(RNHead(), {})
        rnheadpattern = qe.editors.nnmodules.GenericNNModulePattern(qg.fx.quantlib_symbolic_trace, rnheadwithcheckers)
        # create matcher and applier
        finder = qe.editors.nnmodules.GenericGraphMatcher(rnheadpattern)
        applier = RNHeadApplier(rnheadpattern)
        # link pattern, matcher, and applier into the rewriter
        super(RNHeadRewriter, self).__init__('RNHeadRewriter', rnheadpattern, finder, applier)
