# 
# trace.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

import networkx as nx
import os

# import graphs
# import quantlib.editing.graphs.utils

from quantlib.editing.graphs import graphs
from quantlib.editing.graphs import utils


__TRACES_LIBRARY__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.libtraces')
# __TRACES_LIBRARY__ = os.path.join(os.path.expanduser('~'), 'Desktop', 'QuantLab', 'quantlib', 'editing', 'graphs', 'traces', '.libtraces')


def trace_module(library, algorithm, mod, dummy_input):

    # trace graph
    mod.eval()
    onnxgraph = graphs.ONNXGraph(mod, dummy_input)
    G = onnxgraph.nx_graph

    # locate interface nodes
    node_2_partition = nx.get_node_attributes(G, 'bipartite')
    for n in {n for n, onnxnode in onnxgraph.nodes_dict.items() if onnxnode.nobj in set(onnxgraph.jit_graph.inputs()) | set(onnxgraph.jit_graph.outputs())}:
        node_2_partition[n] = graphs.Bipartite.CONTXT
    nx.set_node_attributes(G, node_2_partition, 'partition')

    # store traces and graph picture
    trace_dir = os.path.join(__TRACES_LIBRARY__, library, algorithm, mod.__class__.__name__)
    if not os.path.isdir(trace_dir):
        os.makedirs(trace_dir, exist_ok=True)
    nx.write_gpickle(G, os.path.join(trace_dir, 'networkx'))
    utils.draw_graph(G, trace_dir, 'graphviz')


####################################
## GENERIC PARAMETERS FOR TRACING ##
####################################

_batch_size        = 1
_n_input_channels  = 8
_n_output_channels = 8
_dim1              = 32
_dim2              = 32
_dim3              = 32
_kernel_size       = 3
_stride            = 1
_padding           = 1


#############################
## PYTORCH MODULES TRACING ##
#############################

def trace_pytorch_modules():

    import torch
    import torch.nn as nn

    library = 'PyTorch'

    #####################
    ## AdaptiveAvgPool ##
    #####################
    algorithm = 'AdaptiveAvgPool'

    mod_AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d((int(_dim1 / 4)))
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1)
    trace_module(library, algorithm, mod_AdaptiveAvgPool1d, dummy_input)

    mod_AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((int(_dim1 / 4), int(_dim2 / 4)))
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1, _dim2)
    trace_module(library, algorithm, mod_AdaptiveAvgPool2d, dummy_input)

    mod_AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((int(_dim1 / 4), int(_dim2 / 4), int(_dim3 / 4)))
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1, _dim2, _dim3)
    trace_module(library, algorithm, mod_AdaptiveAvgPool3d, dummy_input)

    ################
    ## torch.view ##
    ################
    algorithm = 'ViewFlatten'

    mod_ViewFlattenNd = graphs.ViewFlattenNd()
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1, _dim2, _dim3)
    trace_module(library, algorithm, mod_ViewFlattenNd, dummy_input)


##############################
## QUANTLIB MODULES TRACING ##
##############################

def trace_quantlib_modules():

    import torch
    import quantlib.algorithms as qa

    library = 'QuantLab'

    #########
    ## STE ##
    #########
    algorithm = 'STE'

    mod_STEActivation = qa.ste.STEActivation()
    dummy_input = torch.ones((_batch_size, _n_input_channels))
    trace_module(library, algorithm, mod_STEActivation, dummy_input)

    #########
    ## INQ ##
    #########
    algorithm = 'INQ'

    mod_INQConv1d = qa.inq.INQConv1d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_inpyut = torch.ones((_batch_size, _n_input_channels, _dim1))
    trace_module(library, algorithm, mod_INQConv1d, dummy_inpyut)

    mod_INQConv2d = qa.inq.INQConv2d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1, _dim2))
    trace_module(library, algorithm, mod_INQConv2d, dummy_input)

    # mod_INQConv3d = qa.inq.INQConv3d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    # dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1, _dim2, _dim3))
    # trace_module(library, algorithm, mod_INQConv3d, dummy_input)

    #########
    ## ANA ##
    #########
    algorithm = 'ANA'

    quantizer_spec = {'nbits': 2, 'signed': True, 'balanced': True, 'eps': 1.0}
    noise_type     = 'uniform'

    mod_ANAActivation = qa.ana.ANAActivation(quantizer_spec, noise_type)
    dummy_input = torch.ones((_batch_size, _n_input_channels))
    trace_module(library, algorithm, mod_ANAActivation, dummy_input)

    mod_ANALinear = qa.ana.ANALinear(quantizer_spec, noise_type, _n_input_channels, _n_output_channels, bias=False)
    dummy_input = torch.ones((_batch_size, _n_input_channels))
    trace_module(library, algorithm, mod_ANALinear, dummy_input)

    mod_ANAConv1d = qa.ana.ANAConv1d(quantizer_spec, noise_type, _n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1))
    trace_module(library, algorithm, mod_ANAConv1d, dummy_input)

    mod_ANAConv2d = qa.ana.ANAConv2d(quantizer_spec, noise_type, _n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1, _dim2))
    trace_module(library, algorithm, mod_ANAConv2d, dummy_input)

    mod_ANAConv3d = qa.ana.ANAConv3d(quantizer_spec, noise_type, _n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1, _dim2, _dim3))
    trace_module(library, algorithm, mod_ANAConv3d, dummy_input)


if __name__ == '__main__':

    if not os.path.isdir(__TRACES_LIBRARY__):
        os.mkdir(__TRACES_LIBRARY__)

    trace_pytorch_modules()
    trace_quantlib_modules()

