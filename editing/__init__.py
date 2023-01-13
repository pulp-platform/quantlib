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

"""A namespace of abstractions to manipulate computational graphs.

A computational graph is a directed acyclic graph (DAG) :math:`G` composed by two types of nodes:
* memory nodes, representing operands and results;
* kernel nodes, representing operations reading from and writing to memory nodes.
This last sentence implies that the graph is bipartite, in that no two memory nodes and no two kernel nodes are directly linked.
In addition, each memory node can be written by zero or one kernel nodes, but it can be read by multiple kernel nodes; i.e., it has at most one incoming arc but it may have any number of of outgoing arcs.
Similarly, each kernel node may read one or more operands but shall write exactly one result; i.e., it has at least one incoming arc and exactly one outgoing arc.
These assumptions hold for low-level, finest-grained representations.
In the context of deep neural networks (DNNs), ONNX provides such a finest-grained representation.
Due to their bipartite nature, I usually think to computational graphs as having coloured nodes:
* red for memory nodes;
* blue for kernel nodes.

However, introducing some semantics allows one to "elevate" the view to higher-level abstractions.
For example, consider the context of deep neural networks.
We notice that:
1. those memory nodes which have no incoming arcs are either inputs to the graph, or parameters of operations (e.g., matrix multiplications, convolutions, and batch-normalisations);
2. each parameter is associated to a specific kernel node, and therefore its associated memory node has exactly one outgoing arc;
3. all the remaining memory nodes have exactly one incoming arc, therefore they are the results of some operations and represent features;
4. some features might have no outgoing arc, which means they are the result of the global computation represented by the graph and are therefore outputs.
We can therefore partition memory nodes into:
* inputs, which have no incoming arcs but one or more outgoing arcs (:math:`1+`);
* parameters, which have no incoming arc and exactly one outgoing arc;
* features, which have exactly one incoming arc and one or more outgoing arcs (:math:`1+`);
* outputs, which have excatly one incoming arc and no outgoing arc.

Observations 2. and 3. allow to uniquely map each parameter (exactly one outgoing arc) and feature (exactly one incoming arc) to exactly one kernel node.
Therefore, given a computational graph :math:`G` whose nodes can be partitioned like this, we can "elevate" it to another graph :math:`H` in the following way.
[...]

I name this process **elevation**.
Elevation preserves directedness and acyclicness.
After evelation, it is no longer possible to distinguish between memory and kernel nodes; however, :math:`H` will still have nodes with no incoming arcs (inputs) and nodes with no outgoing arcs (outputs).
Therefore, the graph :math:`H` is also representing a computation, and is known as a computational graph as well.
I prefer to distinguish the two types of graphs by naming:
* :math:`G` **atomic computational graph**;
* :math:`H` **elevated computational graph**.
I call the inverse operation of elevation **atomisation**.

This vision allows to understand why higher-level abstractions such as ``torch.fx`` IRs and PyTorch ``nn.Module``s represent networks in the way they do.
For instance, in ``torch.fx`` parlance:
* the input nodes of :math:`H` are known as ``placeholders``;
* the remaining nodes are mostly ``call_method`` or ``call_module`` nodes;
* ``output`` nodes represent the elevation of purely sequential graphs (composed of a single "copy" operation and its output memory node) that are attached to the output nodes of the original graph :math:`G`.
These libraries provide functionalities to atomise their elevated computational graphs into atomic computational graphs, such as those in ONNX format.
Such atomisation is obtained by **tracing** the stack of calls issued by PyTorch; therefore, we will use tracing as a synonymous for atomisation.

At a lower level, the aggregation rules are a bit more complicated due to nuances.
Such nuances include:
* the fact that parameters are further partitioned into semantical parameters (e.g., weights, biases, batch-normalisation statistics) and helper parameters (e.g., constants, dropout probabilities);
* the fact that some parameters might be read my multiple atomic operations;
* the fact that not all kernel nodes compute semantical features (e.g., the results of linear, batch-normalisation, and activation operations), but also helper values (e.g., the shape of an array).

"""

from . import graphs
from . import editing
