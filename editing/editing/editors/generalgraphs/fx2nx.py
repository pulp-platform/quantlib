from __future__ import annotations

from collections import OrderedDict
import networkx as nx
import torch.fx as fx


class NXFXGraph(nx.DiGraph):

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
