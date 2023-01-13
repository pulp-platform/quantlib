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

from quantlib.algorithms.qbase.qhparams.qhparams import get_zero_scale
import torch
import torch.fx as fx

from quantlib.editing.editing.editors.nnmodules import NodesMap
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from quantlib.editing.editing.editors.nnmodules import NNModuleApplier


class ActivationRounderApplier(NNModuleApplier):

    def __init__(self, pattern: NNSequentialPattern):
        super(ActivationRounderApplier, self).__init__(pattern)

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        """Modify BN bias and ACT clipping factors to account for ACT rounding.

        This calculates the rounding factor as one half of the activation
        `scale` and then applies it to the BN `bias` directly and ACT `clip_hi`
        hyperparameters.

        """

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_act           = name_to_match_module['act']
        module_bn            = name_to_match_module['bn']

        # modify matched `nn.Module`s in-place
        rounding = module_act.scale.data.detach().clone() * 0.5
        module_bn.bias.data += torch.as_tensor(rounding, device=module_bn.bias.device)
        module_act.clip_hi.data += torch.as_tensor(rounding, device=module_act.clip_hi.device)

        return g
