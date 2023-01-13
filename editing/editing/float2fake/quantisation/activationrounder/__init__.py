# 
# Author(s):
# Francesco Conti <f.conti@unibo.it>
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
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qactivations import PACTReLU, PACTReLU6, PACTLeakyReLU
import torch.nn as nn


from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from quantlib.editing.editing.editors.nnmodules import PathGraphMatcher
from .applier import ActivationRounderApplier
from quantlib.editing.editing.editors.nnmodules import get_rewriter_class
from quantlib.editing.editing.editors import ComposedEditor


# set some constants to reduce code duplication
_N_FEATURES = 1
_BN_KWARGS = {'num_features': _N_FEATURES}
_QRANGE_SPEC_      = (-1, 1)
_QGRANULARITY_SPEC = "per-array"
_QHPARAMSINITSTRATEGY_SPEC = "const"

# map roles to candidate `NNModuleDescription`s that could fit them
roles = Roles([

    # kwargs must mimic the args, kwargs of QLinear, etc. for this to work
    ('bn', Candidates([
        ('BN1d', NNModuleDescription(class_=nn.BatchNorm1d, kwargs=_BN_KWARGS)),
        ('BN2d', NNModuleDescription(class_=nn.BatchNorm2d, kwargs=_BN_KWARGS)),
        ('BN3d', NNModuleDescription(class_=nn.BatchNorm3d, kwargs=_BN_KWARGS)),
    ])),
    ('act', Candidates([
        ('ReLU',      NNModuleDescription(class_=PACTReLU,      kwargs={ 'qrangespec':_QRANGE_SPEC_, 'qgranularityspec': _QGRANULARITY_SPEC, 'qhparamsinitstrategyspec': _QHPARAMSINITSTRATEGY_SPEC, 'inplace': False})),
        ('ReLU6',     NNModuleDescription(class_=PACTReLU6,     kwargs={ 'qrangespec':_QRANGE_SPEC_, 'qgranularityspec': _QGRANULARITY_SPEC, 'qhparamsinitstrategyspec': _QHPARAMSINITSTRATEGY_SPEC, 'inplace': False})),
        ('LeakyReLU', NNModuleDescription(class_=PACTLeakyReLU, kwargs={ 'qrangespec':_QRANGE_SPEC_, 'qgranularityspec': _QGRANULARITY_SPEC, 'qhparamsinitstrategyspec': _QHPARAMSINITSTRATEGY_SPEC, 'inplace': False}))
    ])),

])

# define target patterns
admissible_screenplays = [
    ('BN1d', 'ReLU'),
    ('BN2d', 'ReLU'),
    ('BN3d', 'ReLU'),
    ('BN1d', 'ReLU6'),
    ('BN2d', 'ReLU6'),
    ('BN3d', 'ReLU6'),
    ('BN1d', 'LeakyReLU'),
    ('BN2d', 'LeakyReLU'),
    ('BN3d', 'LeakyReLU'),
]

# programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
namespace = OrderedDict([])
for name, pattern in generate_named_patterns(roles, admissible_screenplays):
    class_name = name + 'ActRounder'
    class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, ActivationRounderApplier)
    globals()[class_name] = class_   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    namespace[class_name] = class_   # keep track of the new class so that we can later create the generic bias folder


# create the general-purpose `ActivationRounder`
class ActivationRounder(ComposedEditor):
    """Activation rounder."""
    def __init__(self):
        super(ActivationRounder, self).__init__([class_() for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['ActivationRounder']
