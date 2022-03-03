# 
# draw.py
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

from collections import namedtuple
import networkx as nx
import graphviz as gv

from ...graphs import graphs


GVNodeAppearance = namedtuple('GVNodeAppearance', ['fontsize', 'shape', 'height', 'width', 'color', 'fillcolor'])


def _get_styles():

    _styles = {
        graphs.Bipartite.KERNEL:
            GVNodeAppearance(fontsize='8', shape='circle', height='2.0', width='2.0',
                             color='cornflowerblue', fillcolor='cornflowerblue'),
        graphs.Bipartite.MEMORY:
            GVNodeAppearance(fontsize='8', shape='square', height='1.2', width='1.2',
                             color='brown2', fillcolor='brown2')
    }

    return _styles


def draw_graph(G, save_dir, filename, node_2_label=None):

    # map nodes to labels and graphic styles
    partition_2_style = _get_styles()
    if (node_2_label is None) or (set(G.nodes) != set(node_2_label.keys())):
        node_2_label = {n: G.nodes[n]['type'] for n in G.nodes}

    # build GraphViz graph
    gvG = gv.Digraph(comment=filename)
    for n, p in nx.get_node_attributes(G, 'bipartite').items():
        gvG.node(n, node_2_label[n], **partition_2_style[p]._asdict(), style='filled')
    for e in G.edges:
        gvG.edge(e[0], e[1])

    gvG.render(directory=save_dir, filename=filename)
