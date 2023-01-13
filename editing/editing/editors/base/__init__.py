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

"""This package implements the hierachy of base classes of QuantLib's graph
editing machinery.

Edits are partitioned into basic edits and composed edits. Basic edits are
further partitioned into annotations (which add attributes and other semantic
information but do not modify functional graph information) and rewritings
(which can modify functional graph information, and even change the topology
of the graph).

"""

#                      Editor                                                     #
#                        /\                                                       #
#                       /  \                                                      #
#          BaseEditor _/    \_ ComposedEditor                                     #
#              /\                                                                 #
#             /  \                                                                #
#  Annotator_/    \_ Rewriter                                                     #
#                    |_ ApplicationPoint + Context = ApplicationPointWithContext  #
#                    |_ Finder                                                    #
#                    |_ Applier                                                   #

from .annotator import Annotator
from .rewriter import ApplicationPoint, Finder, Applier, Rewriter
from .composededitor import ComposedEditor
