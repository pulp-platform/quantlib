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
import torch
import torch.nn as nn

from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns, get_rewriter_class
from .finder import LinearOpIntegeriserMatcher
from .applier import LinearOpIntegeriserApplier
from quantlib.editing.editing.editors.nnmodules import NNModuleRewriter
from quantlib.editing.editing.editors import ComposedEditor
from quantlib.editing.graphs.nn import EpsTunnel


# set some constants to reduce code duplication
_N_FEATURES = 1
_HAS_BIAS = False
_KERNEL_SIZE = 1
_CONV_KWARGS = {'in_channels': _N_FEATURES, 'out_channels': _N_FEATURES, 'kernel_size': _KERNEL_SIZE, 'bias': _HAS_BIAS}
_LINEAROP_CHECKERS = (lambda m: m.bias is None,)
_EPS = torch.Tensor([1.0])
_EPS_KWARGS = {'eps': _EPS}

# map roles to candidate `NNModuleDescription`s that could fit them
roles = Roles([

    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),

    ('linear', Candidates([
        ('QLinear', NNModuleDescription(class_=nn.Linear, kwargs={'in_features': _N_FEATURES, 'out_features': _N_FEATURES, 'bias': _HAS_BIAS}, checkers=_LINEAROP_CHECKERS)),
        ('QConv1d', NNModuleDescription(class_=nn.Conv1d, kwargs=_CONV_KWARGS, checkers=_LINEAROP_CHECKERS)),
        ('QConv2d', NNModuleDescription(class_=nn.Conv2d, kwargs=_CONV_KWARGS, checkers=_LINEAROP_CHECKERS)),
        ('QConv3d', NNModuleDescription(class_=nn.Conv3d, kwargs=_CONV_KWARGS, checkers=_LINEAROP_CHECKERS)),
    ])),

    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),
])

# define target patterns
admissible_screenplays = list(roles.all_screenplays)

# programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
namespace = OrderedDict([])
for name, pattern in generate_named_patterns(roles, admissible_screenplays):
    class_name = name + 'Integeriser'
    class_ = get_rewriter_class(class_name, pattern, LinearOpIntegeriserMatcher, LinearOpIntegeriserApplier)
    globals()[class_name] = class_   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    namespace[class_name] = class_   # keep track of the new class so that we can later create the generic integeriser of linear opeartions


# create the general-purpose `LinearOpIntegeriser`
class LinearOpIntegeriser(ComposedEditor):
    def __init__(self):
        super(LinearOpIntegeriser, self).__init__([class_() for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['LinearOpIntegeriser']
