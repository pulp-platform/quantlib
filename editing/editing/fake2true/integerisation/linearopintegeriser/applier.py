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

from quantlib.editing.editing.editors.nnmodules import NodesMap
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from quantlib.editing.editing.editors.nnmodules import NNModuleApplier
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.algorithms.qmodules.qmodules.qlinears import SUPPORTED_LINEAR_FPMODULES


class LinearOpIntegeriserApplier(NNModuleApplier):

    def __init__(self, pattern: NNSequentialPattern):
        super(LinearOpIntegeriserApplier, self).__init__(pattern)

    @staticmethod
    def from_qlinear(qlinear: _QModule) -> nn.Module:
        """Return an ``nn.Module`` implementing a linear operation with
        integerised parameters.
        """
        # TODO: should I offload the responsibility of computing the true-quantised `nn.Module` to `_QLinear`?
        if not isinstance(qlinear, SUPPORTED_LINEAR_FPMODULES):
            raise TypeError
        
        if isinstance(qlinear, nn.Linear):
            class_ = nn.Linear
            new_module = class_(in_features=qlinear.in_features,
                                out_features=qlinear.out_features,
                                bias=True)
            if not qlinear.bias:
                with torch.no_grad():
                    new_module.bias[:] = 0

        elif isinstance(qlinear, (nn.Conv1d, nn.Conv2d, nn.Conv3d,)):
            if isinstance(qlinear, nn.Conv1d):
                class_ = nn.Conv1d
            elif isinstance(qlinear, nn.Conv2d):
                class_ = nn.Conv2d
            else:  # `isinstance(qlinear, nn.Conv3d)`
                class_ = nn.Conv3d
            new_module = class_(in_channels=qlinear.in_channels,
                                out_channels=qlinear.out_channels,
                                kernel_size=qlinear.kernel_size,
                                stride=qlinear.stride,
                                padding=qlinear.padding,
                                dilation=qlinear.dilation,
                                groups=qlinear.groups,
                                bias=qlinear.bias)

        else:
            raise RuntimeError

        iweight = torch.round(qlinear.qweight.data.clone().detach() / qlinear.scale.data.clone().detach())  # integerised parameters
        new_module.weight.data = iweight

        return new_module

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_eps_in  = name_to_match_node['eps_in']
        node_linear  = name_to_match_node['linear']
        node_eps_out = name_to_match_node['eps_out']

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in  = name_to_match_module['eps_in']
        module_linear  = name_to_match_module['linear']
        module_eps_out = name_to_match_module['eps_out']

        # create the integerised linear operation
        new_target = id_
        new_module = LinearOpIntegeriserApplier.from_qlinear(module_linear)

        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_linear, new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))

        # ...and delete the old operation
        g.delete_submodule(node_linear.target)
        g.graph.erase_node(node_linear)

        return g
