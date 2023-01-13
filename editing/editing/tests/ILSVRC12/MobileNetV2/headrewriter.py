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


class MNv2Head(nn.Module):

    def __init__(self):
        super(MNv2Head, self).__init__()
        self.eps = qg.nn.EpsTunnel(torch.Tensor([1.0]))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.drp = nn.Dropout(0.2)
        self.lin = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eps(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.drp(x)
        x = self.lin(x)
        return x


class MNv2HeadApplier(qe.editors.nnmodules.NNModuleApplier):

    def __init__(self, mnv2headpattern: qe.editors.nnmodules.GenericNNModulePattern):
        super(MNv2HeadApplier, self).__init__(mnv2headpattern)

    def _apply(self, g: fx.GraphModule, ap: qe.editors.nnmodules.NodesMap, id_: str) -> fx.GraphModule:

        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_drp = name_to_match_node['drp']
        node_lin = name_to_match_node['lin']

        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps = name_to_match_module['eps']
        module_lin = name_to_match_module['lin']

        assert module_eps.eps_out.numel() == 1
        assert len(node_drp.all_input_nodes) == 1
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
        drp_input = next(iter(node_drp.all_input_nodes))
        with g.graph.inserting_after(drp_input):
            new_node = g.graph.call_module(new_target, args=(drp_input,))
        node_lin.replace_all_uses_with(new_node)

        module_eps.set_eps_out(torch.ones_like(module_eps.eps_out))

        # ...and delete the old operations
        g.delete_submodule(node_lin.target)
        g.graph.erase_node(node_lin)
        g.delete_submodule(node_drp.target)
        g.graph.erase_node(node_drp)

        return g


class MNv2HeadRewriter(qe.editors.nnmodules.NNModuleRewriter):

    def __init__(self):
        # create pattern
        mnv2headwithcheckers = qe.editors.nnmodules.NNModuleWithCheckers(MNv2Head(), {})
        mnv2headpattern = qe.editors.nnmodules.GenericNNModulePattern(qg.fx.quantlib_symbolic_trace, mnv2headwithcheckers)
        # create matcher and applier
        finder = qe.editors.nnmodules.GenericGraphMatcher(mnv2headpattern)
        applier = MNv2HeadApplier(mnv2headpattern)
        # link pattern, matcher, and applier into the rewriter
        super(MNv2HeadRewriter, self).__init__('MNv2HeadRewriter', mnv2headpattern, finder, applier)
