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

from __future__ import annotations

from collections import OrderedDict
import networkx as nx
import torch.fx as fx


class NXFXGraph(nx.DiGraph):

    # TODO: accept as nodes only those objects that have an `"fx"` attribute which is an `fx.Node` object

    def __init__(self):
        super(NXFXGraph, self).__init__()

    @staticmethod
    def from_fx_graph(F: fx.Graph) -> NXFXGraph:

        N = NXFXGraph()

        # create the vertex set of the `NXFXGraph`
        VN_2_VF = OrderedDict(enumerate(F.nodes))
        N.add_nodes_from(VN_2_VF.keys())

        # extract the connectivity of the `fx.Graph`
        EF = set()
        for n in F.nodes:
            for p in n.all_input_nodes:
                EF.add((p, n))
            for s in n.users:
                EF.add((n, s))

        # transfer the connectivity to the `NXFXGraph`
        VF_2_VN = {v: k for k, v in VN_2_VF.items()}
        EN = set(map(lambda arc: (VF_2_VN[arc[0]], VF_2_VN[arc[1]]), EF))
        N.add_edges_from(EN)

        # This is where `NXFXGraph`s differ from base `nx.DiGraph`s: each node
        # of an `NXFXGraph` is bijectively associated with an `fx.Node` of the
        # originating `fx.Graph`.
        nx.set_node_attributes(N, VN_2_VF, name='fx')

        return N
