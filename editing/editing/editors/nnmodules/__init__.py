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

"""This package implements the abstractions required to rewrite ``fx.Graph``s
obtained by tracing ``nn.Module``s.

Given an input ``fx.Graph``, an ``NNModuleRewriter`` can identify its
sub-graphs that are isomorphic to a given pattern ``fx.Graph`` and replace
them with specified target ``fx.Graph``s. The logic to identify sub-graphs is
implemented by ``NNModuleMatcher``s, whereas the logic to replace the matches
is implemented by ``NNModuleApplier``s.

In this package, a match is implemented as a ``NodesMap`` data structure, a
dictionary mapping ``fx.Node``s in the pattern ``fx.Graph`` to ``fx.Node``s in
the input `fx.Graph``.

Both the ``NNModuleMatcher`` and ``NNModuleApplier`` associated with a given
``NNModuleRewriter`` are intended to be engineered accordingly to a specific
``NNModulePattern`` object. An ``NNModulePattern`` is an ``fx.GraphModule``
exposing dictionaries to give symbolic (i.e., name-based) access to several
structures:
* its composing ``nn.Module``s;
* the ``nn.Module``s associated with the ``fx.Node``s in a matched sub-graph;
* collections of functions (``Checkers``) that can be used to verify whether
  a given ``nn.Module`` satisfies given properties.
Users can create ``NNModulePattern``s by tracing ``NNModuleWithCheckers``
objects. An ``NNModuleWithCheckers`` pairs an ``nn.Module`` with per-module
collections of ``Checker``s.

To accelerate the identification of matching sub-graphs, we resort to a simple
back-tracking logic when considering ``NNModulePattern``s created from
``nn.Sequential`` objects, as opposed to solving the sub-graph isomorphism
problem when considering generic ``nn.Module``s.

We provide developers with functionalities to generate entire families of
``NNSequentialPattern``s, as well as families of ``NNModuleRewriter``s.

"""

#                                                                                       #
#                 NNModuleWithCheckers                                                  #
#                          |                                                            #
#                          |                                                            #
#                          |                                                            #
#                  (NNModulePattern) -----------------------------------------+         #
#                          /\                                                 |         #
#  NNSequentialPattern  _/ |  \_  GenericNNModulePattern                      |         #
#          |               |                |                                 |         #
#          |               |                |                                 |         #
#          |       (NNModuleMatcher)        |                          NNModuleApplier  #
#          |               /\               |                                 |         #
#   PathGraphMatcher  ___/ |  \___ GenericGraphMatcher                        |         #
#                          |                                                  |         #
#                          |                                                  |         #
#                          |                                                  |         #
#                          +----------------- NNModuleRewriter ---------------+         #
#                                                                                       #

from .applicationpoint import NodesMap
from .pattern import NNModuleWithCheckers
from .pattern import NNSequentialPattern, NNModuleDescription, Candidates, Roles, generate_named_patterns
from .pattern import GenericNNModulePattern
from .finder import PathGraphMatcher
from .finder import GenericGraphMatcher
from .applier import NNModuleApplier
from .rewriter import NNModuleRewriter, get_rewriter_class
