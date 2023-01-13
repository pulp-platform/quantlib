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
import torch.nn as nn
import torch.fx as fx

from ..qdescription import QDescription, QDescriptionSpecType, resolve_qdescriptionspec
from quantlib.editing.editing.editors.nnmodules import NodesMap
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from quantlib.editing.editing.editors.nnmodules import NNModuleApplier


class QuantiserInterposerApplier(NNModuleApplier):

    def __init__(self,
                 qdescriptionspec: QDescriptionSpecType,
                 pattern:          NNSequentialPattern):

        qdescription = resolve_qdescriptionspec(qdescriptionspec)

        super(QuantiserInterposerApplier, self).__init__(pattern)
        self._qdescription = qdescription

    @property
    def qdescription(self) -> QDescription:
        return self._qdescription

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        """Insert a quantiser between two linear opeartions."""

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_pre  = name_to_match_node['linear_pre']
        node_post = name_to_match_node['linear_post']

        # create the new quantiser
        new_target = id_
        qgranularityspec, qrangespec, qhparamsinitstrategyspec, (mapping, kwargs) = copy.deepcopy(self.qdescription)
        new_module = mapping[nn.Identity](qrangespec=qrangespec,
                                          qgranularityspec=qgranularityspec,
                                          qhparamsinitstrategyspec=qhparamsinitstrategyspec,
                                          **kwargs)

        # add the quantiser to the graph (interposing it between the two linear nodes)
        # We want that after the rewriting each user of `node_pre` reads the
        # output of `new_node` instead; however, in the intermediate state,
        # `new_node` will itself be a user of `node_pre`. Therefore, we need
        # to determine who these users are before `new_node` becomes one of
        # them.
        downstream_nodes = list(node_pre.users)
        assert node_post in downstream_nodes
        # rewrite the graph
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_pre):
            new_node = g.graph.call_module(new_target, args=(node_pre,))
        for u in downstream_nodes:
            u.replace_input_with(node_pre, new_node)

        return g
