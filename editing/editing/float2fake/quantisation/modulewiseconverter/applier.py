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

import copy
import torch.fx as fx

from .applicationpoint import PartitionId, NodeWithPartition
from .modulewisedescription import ModuleWiseDescription
from .modulewisedescription import NameToModule
from quantlib.editing.editing.editors import Applier


class ModuleWiseReplacer(Applier):

    def __init__(self, modulewisedescription: ModuleWiseDescription):

        if not isinstance(modulewisedescription, ModuleWiseDescription):
            raise TypeError

        super(ModuleWiseReplacer, self).__init__()
        self._modulewisedescription = modulewisedescription

    @property
    def modulewisedescription(self) -> ModuleWiseDescription:
        return self._modulewisedescription

    def _apply(self, g: fx.GraphModule, ap: NodeWithPartition, id_: str) -> fx.GraphModule:

        # retrieve the old (floating-point) `nn.Module`
        node: fx.Node = ap.node
        fpmodule = g.get_submodule(target=node.target)

        # retrieve the quantisation instructions (`QDescription`) for the information
        partition_id: PartitionId = ap.id_
        _, qspecification = self.modulewisedescription[partition_id]

        # create the new (fake-quantised) `nn.Module`
        qgranularity, qrange, qhparamsinitstrategy, (mapping, kwargs) = copy.deepcopy(qspecification)  # ensure that different instances use different objects
        qmodule_class = mapping[type(fpmodule)]
        qmodule = qmodule_class.from_fp_module(fpmodule, qrange, qgranularity, qhparamsinitstrategy, **kwargs)  # TODO: if `fpmodule` is in evaluation state, will `fqmodule` also be in such state?

        # insert the fake-quantised module into the graph (note that we do not use any `torch.fx` rewriting here)
        name_to_module = NameToModule(g.named_modules())
        name_to_module[node.target] = qmodule
        path_to_parent, child = NameToModule.split_path_to_target(node.target)
        setattr(name_to_module[path_to_parent], child, qmodule)  # https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L44

        return g
