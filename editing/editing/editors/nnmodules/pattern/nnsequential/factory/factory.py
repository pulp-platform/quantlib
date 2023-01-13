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

from collections import OrderedDict
import torch.nn as nn
from typing import NamedTuple, List

from .roles import Roles, Screenplay
from ..pattern import NNSequentialPattern
from ...base import NNModuleWithCheckers
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


PatternName = str


class NamedNNSequentialPattern(NamedTuple):
    name:    PatternName
    pattern: NNSequentialPattern


def generate_named_patterns(roles: Roles,
                            candidate_screeplays: List[Screenplay]) -> List[NamedNNSequentialPattern]:
    """A utility to generate ``NNSequentialPattern``s programmatically."""

    named_patterns: List[NamedNNSequentialPattern] = []

    for screenplay in candidate_screeplays:

        # raise an error if the screenplay is not valid
        roles.check_screenplay(screenplay)

        # we traverse the screenplay to fill in all roles
        names: List[str] = []
        modules = OrderedDict([])  # we will use this container to create an `nn.Sequential`
        name_to_checkers = OrderedDict([])

        for role, candidate in zip(roles.roles, screenplay):

            # retrive candidate description
            desc = roles[role][candidate]

            if desc is not None:
                names.append(candidate)
                modules[role] = desc.class_(**desc.kwargs)
                name_to_checkers[role] = desc.checkers

        # create the `NNModuleWithChecker`s
        nnsequential = nn.Sequential(modules)
        nnmodulewithcheckers = NNModuleWithCheckers(module=nnsequential, name_to_checkers=name_to_checkers)

        # create the pattern
        pattern_name = ''.join(names)
        pattern = NNSequentialPattern(symbolic_trace_fn=quantlib_symbolic_trace, nnmodulewithcheckersspec=nnmodulewithcheckers)
        named_pattern = NamedNNSequentialPattern(name=pattern_name, pattern=pattern)

        named_patterns.append(named_pattern)

    return named_patterns
