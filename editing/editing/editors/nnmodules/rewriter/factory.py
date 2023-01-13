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

from typing import Union, Type

from ..pattern import NNSequentialPattern, GenericNNModulePattern
from ..finder import PathGraphMatcher, GenericGraphMatcher
from ..applier import NNModuleApplier
from .rewriter import NNModuleRewriter


def get_rewriter_class(name:          str,
                       pattern:       Union[NNSequentialPattern, GenericNNModulePattern],
                       finder_class:  Type[Union[PathGraphMatcher, GenericGraphMatcher]],
                       applier_class: Type[NNModuleApplier]) -> Type[NNModuleRewriter]:
    """A utility to generate ``NNModuleRewriter`` classes programmatically."""

    # validate input types
    if not isinstance(name, str):
        raise TypeError
    if not isinstance(pattern, (NNSequentialPattern, GenericNNModulePattern,)):
        raise TypeError
    if not isinstance(finder_class, (type(PathGraphMatcher), type(GenericGraphMatcher),)):
        raise TypeError
    if not isinstance(applier_class, type(NNModuleApplier)):
        raise TypeError

    # create the `Finder`
    finder = finder_class(pattern)

    # create the `Applier`
    applier = applier_class(pattern)

    # sub-class `Rewriter` around these specific `Finder` and `Applier`
    def __init__(self_):
        NNModuleRewriter.__init__(self_, name, pattern, finder, applier)

    rewriter_class = type(name, (NNModuleRewriter,), {'__init__': __init__})

    return rewriter_class
