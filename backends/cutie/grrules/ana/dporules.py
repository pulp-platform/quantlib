# 
# dporules.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zuerich.
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

import itertools
from collections import OrderedDict
import torch
import torch.nn as nn
import networkx as nx

from .lutactivation import LUTActivation

from .folding import fold_anaact_anaconv2d_bn2d_anaact, fold_anaact_analinear_bn1d_anaact
from quantlib.editing.graphs.graphs import Bipartite, PyTorchNode, __NODE_ID_FORMAT__
from quantlib.editing.graphs.grrules.dporules import DPORule
from quantlib.editing.graphs.grrules import Seeker
import quantlib.editing.graphs as qg

import quantlib.algorithms as qa


class FoldANAConvBNANAActRule(DPORule):

    def __init__(self, lut_entry_bits=16):

        self._lut_entry_bits = lut_entry_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPin':  qg.graphs.HelperInput.__name__})
        K_types.update({'HPTin': qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'ANAConv':   qa.ana.ANAConv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ANAActout': qa.ana.ANAActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'TWConv': nn.Conv2d.__name__})
        RK_types.update({'LUTAct': LUTActivation.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs  = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: Bipartite.KERNEL for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: Bipartite.KERNEL for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FConvBNANA', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        mconv2d = nodes_dict[g_L2H['/'.join(['L-term', 'ANAConv'])]].nobj
        mbn2d   = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        manaout = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActout'])]].nobj

        # fold
        tau, weight = fold_anaact_anaconv2d_bn2d_anaact(torch.Tensor([1.0]),
                                                        mconv2d.eps, mconv2d.weight_maybe_quant,
                                                        mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight,
                                                        mbn2d.bias,
                                                        manaout.eps,
                                                        manaout.thresholds,
                                                        ceiltau=False)

        # build the new modules
        mtwconv = nn.Conv2d(mconv2d.in_channels, mconv2d.out_channels, mconv2d.kernel_size,
                            stride=mconv2d.stride, padding=mconv2d.padding, dilation=mconv2d.dilation,
                            groups=mconv2d.groups,
                            bias=mconv2d.bias is not None).to(torch.device('cpu'))
        mtwconv.weight.data = weight

        mlutact = LUTActivation(tau, manaout.quant_levels)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWConv'])]] = PyTorchNode(mtwconv)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'LUTAct'])]] = PyTorchNode(mlutact)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}  # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}  # Occurence of core template
        HI = G.subgraph(VHI)  # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)  # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode)  # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:  # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in
                              self.F_K2RK[vK]})  # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in
                              self.F_RK2K[vK]})  # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperInput.__name__:
                pass
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldANAActANAConvBNANAActTypeARule(DPORule):  # w/o max pooling

    def __init__(self, lut_entry_bits=16):

        self._lut_entry_bits = lut_entry_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPTin': qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'ANAActin': qa.ana.ANAActivation.__name__})
        LK_types.update({'ANAConv': qa.ana.ANAConv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ANAActout': qa.ana.ANAActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'TWConv': nn.Conv2d.__name__})
        RK_types.update({'LUTAct': LUTActivation.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: Bipartite.KERNEL for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: Bipartite.KERNEL for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FANABNANATA', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        manain = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActin'])]].nobj
        mconv2d = nodes_dict[g_L2H['/'.join(['L-term', 'ANAConv'])]].nobj
        mbn2d = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        manaout = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActout'])]].nobj

        # fold
        tau, weight = fold_anaact_anaconv2d_bn2d_anaact(manain.eps,
                                                        mconv2d.eps, mconv2d.weight_maybe_quant,
                                                        mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight,
                                                        mbn2d.bias,
                                                        manaout.eps,
                                                        manaout.thresholds)

        # build the new modules
        mtwconv = nn.Conv2d(mconv2d.in_channels, mconv2d.out_channels, mconv2d.kernel_size,
                            stride=mconv2d.stride, padding=mconv2d.padding, dilation=mconv2d.dilation,
                            groups=mconv2d.groups,
                            bias=mconv2d.bias is not None).to(torch.device('cpu'))
        mtwconv.weight.data = weight

        mlutact = LUTActivation(tau, manaout.quant_levels)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWConv'])]] = PyTorchNode(mtwconv)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'LUTAct'])]] = PyTorchNode(mlutact)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}  # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}  # Occurence of core template
        HI = G.subgraph(VHI)  # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)  # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode)  # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:  # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in
                              self.F_K2RK[vK]})  # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in
                              self.F_RK2K[vK]})  # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldANAActANAConvBNANAActTypeBRule(DPORule):  # w/ max pooling

    def __init__(self, lut_entry_bits=16):

        self._lut_entry_bits = lut_entry_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPTin': qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'ANAActin': qa.ana.ANAActivation.__name__})
        LK_types.update({'MaxPool': nn.MaxPool2d.__name__})
        LK_types.update({'ANAConv': qa.ana.ANAConv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ANAActout': qa.ana.ANAActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'MaxPool': nn.MaxPool2d.__name__})
        RK_types.update({'TWConv': nn.Conv2d.__name__})
        RK_types.update({'LUTAct': LUTActivation.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: Bipartite.KERNEL for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: Bipartite.KERNEL for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FANABNANATB', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        manain = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActin'])]].nobj
        mmxpold = nodes_dict[g_L2H['/'.join(['L-term', 'MaxPool'])]].nobj
        mconv2d = nodes_dict[g_L2H['/'.join(['L-term', 'ANAConv'])]].nobj
        mbn2d = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        manaout = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActout'])]].nobj

        # fold
        tau, weight = fold_anaact_anaconv2d_bn2d_anaact(manain.eps,
                                                        mconv2d.eps, mconv2d.weight_maybe_quant,
                                                        mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight,
                                                        mbn2d.bias,
                                                        manaout.eps,
                                                        manaout.thresholds)

        # build the new modules
        mmxpnew = nn.MaxPool2d(kernel_size=mmxpold.kernel_size, stride=mmxpold.stride, padding=mmxpold.padding)

        mtwconv = nn.Conv2d(mconv2d.in_channels, mconv2d.out_channels, mconv2d.kernel_size,
                            stride=mconv2d.stride, padding=mconv2d.padding, dilation=mconv2d.dilation,
                            groups=mconv2d.groups,
                            bias=mconv2d.bias is not None).to(torch.device('cpu'))
        mtwconv.weight.data = weight

        mlutact = LUTActivation(tau, manaout.quant_levels)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'MaxPool'])]] = PyTorchNode(mmxpnew)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWConv'])]] = PyTorchNode(mtwconv)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'LUTAct'])]] = PyTorchNode(mlutact)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}  # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}  # Occurence of core template
        HI = G.subgraph(VHI)  # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)  # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode)  # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:  # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in
                              self.F_K2RK[vK]})  # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in
                              self.F_RK2K[vK]})  # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldANAActANALinearBNANAActTypeARule(DPORule):  # w/o pooling layers

    def __init__(self, lut_entry_bits=16):

        self._lut_entry_bits = lut_entry_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPTin':  qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'ANAActin':  qa.ana.ANAActivation.__name__})
        LK_types.update({'ANALinear': qa.ana.ANALinear.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm1d.__name__})
        LK_types.update({'ANAActout': qa.ana.ANAActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'TWLinear': nn.Linear.__name__})
        RK_types.update({'LUTAct':   LUTActivation.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: Bipartite.KERNEL for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: Bipartite.KERNEL for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FANABNANALinTA', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        manain = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActin'])]].nobj
        mlinear = nodes_dict[g_L2H['/'.join(['L-term', 'ANALinear'])]].nobj
        mbn1d = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        manaout = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActout'])]].nobj

        # fold
        tau, weight = fold_anaact_analinear_bn1d_anaact(manain.eps,
                                                        mlinear.eps, mlinear.weight_maybe_quant,
                                                        mbn1d.running_mean, mbn1d.running_var, mbn1d.eps, mbn1d.weight,
                                                        mbn1d.bias,
                                                        manaout.eps,
                                                        manaout.thresholds)

        # build the new modules
        mtwlinear = nn.Linear(mlinear.in_features, mlinear.out_features,
                              bias=mlinear.bias is not None).to(torch.device('cpu'))
        mtwlinear.weight.data = weight

        mlutact = LUTActivation(tau, manaout.quant_levels)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWLinear'])]] = PyTorchNode(mtwlinear)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'LUTAct'])]]   = PyTorchNode(mlutact)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}  # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}  # Occurence of core template
        HI = G.subgraph(VHI)  # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)  # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode)  # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:  # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in
                              self.F_K2RK[vK]})  # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in
                              self.F_RK2K[vK]})  # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldANAActANALinearBNANAActTypeBRule(DPORule):  # w/ pooling layers

    def __init__(self, lut_entry_bits=16):

        self._lut_entry_bits = lut_entry_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPTin': qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'ANAActin': qa.ana.ANAActivation.__name__})
        LK_types.update({'MaxPool': nn.MaxPool2d.__name__})
        LK_types.update({'AvgPool': nn.AdaptiveAvgPool2d.__name__})
        LK_types.update({'ViewFlattenNd': qg.graphs.modules.ViewFlattenNd.__name__})
        LK_types.update({'ANALinear': qa.ana.ANALinear.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm1d.__name__})
        LK_types.update({'ANAActout': qa.ana.ANAActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'MaxPool': nn.MaxPool2d.__name__})
        RK_types.update({'TWLinear': nn.Linear.__name__})
        RK_types.update({'LUTAct': LUTActivation.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: Bipartite.KERNEL for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: Bipartite.KERNEL for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FANABNANALinTB', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        manain = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActin'])]].nobj
        mmxpold = nodes_dict[g_L2H['/'.join(['L-term', 'MaxPool'])]].nobj
        mlinear = nodes_dict[g_L2H['/'.join(['L-term', 'ANALinear'])]].nobj
        mbn1d = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        manaout = nodes_dict[g_L2H['/'.join(['L-term', 'ANAActout'])]].nobj

        # fold
        tau, weight = fold_anaact_analinear_bn1d_anaact(manain.eps,
                                                        mlinear.eps, mlinear.weight_maybe_quant,
                                                        mbn1d.running_mean, mbn1d.running_var, mbn1d.eps, mbn1d.weight,
                                                        mbn1d.bias,
                                                        manaout.eps,
                                                        manaout.thresholds)

        # build the new modules
        mmxpnew = nn.MaxPool2d(kernel_size=mmxpold.kernel_size, stride=mmxpold.stride, padding=mmxpold.padding)

        mtwlinear = nn.Linear(mlinear.in_features, mlinear.out_features,
                              bias=mlinear.bias is not None).to(torch.device('cpu'))
        mtwlinear.weight.data = weight

        mlutact = LUTActivation(tau, manaout.quant_levels)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'MaxPool'])]] = PyTorchNode(mmxpnew)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWLinear'])]] = PyTorchNode(mtwlinear)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'LUTAct'])]] = PyTorchNode(mlutact)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}  # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}  # Occurence of core template
        HI = G.subgraph(VHI)  # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)  # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode)  # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:  # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in
                              self.F_K2RK[vK]})  # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in
                              self.F_RK2K[vK]})  # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldANAActLinearRule(DPORule):

    def __init__(self, lut_entry_bits=16):

        self._lut_entry_bits = lut_entry_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPout': qg.graphs.HelperOutput.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'ANAAct': qa.ana.ANAActivation.__name__})
        LK_types.update({'Linear': nn.Linear.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'Linear': nn.Linear.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: Bipartite.KERNEL for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        ### WARNING! if R\K has only one node, this initialisation will fail!
        self.RK.add_nodes_from(RK_node_IDs)
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: Bipartite.KERNEL for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FANAActLinear', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        manain = nodes_dict[g_L2H['/'.join(['L-term', 'ANAAct'])]].nobj
        mlinearold = nodes_dict[g_L2H['/'.join(['L-term', 'Linear'])]].nobj

        # fold
        weight = manain.eps.item() * mlinearold.weight.data

        # build the new modules
        mlinearnew = nn.Linear(mlinearold.in_features, mlinearold.out_features,
                               bias=mlinearold.bias is not None).to(torch.device('cpu'))
        mlinearnew.weight.data = weight
        mlinearnew.bias.data = mlinearold.bias.data

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'Linear'])]] = PyTorchNode(mlinearnew)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}  # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}  # Occurence of core template
        HI = G.subgraph(VHI)  # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)  # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode)  # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:  # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in
                              self.F_K2RK[vK]})  # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in
                              self.F_RK2K[vK]})  # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperOutput.__name__:
                pass
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs
