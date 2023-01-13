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
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qlinears import PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d
import torch.nn as nn


from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from quantlib.editing.editing.editors.nnmodules import PathGraphMatcher
from .applier import WeightRounderApplier
from quantlib.editing.editing.editors.nnmodules import get_rewriter_class
from quantlib.editing.editing.editors import ComposedEditor

_QRANGE_SPEC_      = (-1, 1)
_QGRANULARITY_SPEC = "per-array"
_QHPARAMSINITSTRATEGY_SPEC = ('const', {'a': 0.0, 'b': 6.0})
_IN_FEATURES = 1
_OUT_FEATURES = 1
_KERNEL_SIZE = 1
_BIAS = True
_CONV_KWARGS = {'qrangespec':_QRANGE_SPEC_, 'qgranularityspec': _QGRANULARITY_SPEC, 'qhparamsinitstrategyspec': _QHPARAMSINITSTRATEGY_SPEC, 'in_channels': _IN_FEATURES, 'out_channels': _OUT_FEATURES, 'kernel_size': _KERNEL_SIZE, 'bias': _BIAS}
_LIN_KWARGS  = {'qrangespec':_QRANGE_SPEC_, 'qgranularityspec': _QGRANULARITY_SPEC, 'qhparamsinitstrategyspec': _QHPARAMSINITSTRATEGY_SPEC, 'in_features': _IN_FEATURES, 'out_features': _OUT_FEATURES, 'bias': _BIAS}

# map roles to candidate `NNModuleDescription`s that could fit them
roles = Roles([

    # kwargs must mimic the args, kwargs of QLinear, etc. for this to work
    ('linear', Candidates([
        ('Linear', NNModuleDescription(class_=PACTLinear, kwargs=_LIN_KWARGS)),
        ('Conv1d', NNModuleDescription(class_=PACTConv1d, kwargs=_CONV_KWARGS)),
        ('Conv2d', NNModuleDescription(class_=PACTConv2d, kwargs=_CONV_KWARGS)),
        ('Conv3d', NNModuleDescription(class_=PACTConv3d, kwargs=_CONV_KWARGS)),
    ])),

])

# define target patterns
admissible_screenplays = [
    ('Linear',),
    ('Conv1d',),
    ('Conv2d',),
    ('Conv3d',)
]

# programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
namespace = OrderedDict([])
for name, pattern in generate_named_patterns(roles, admissible_screenplays):
    class_name = name + 'WeightRounder'
    class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, WeightRounderApplier)
    globals()[class_name] = class_   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    namespace[class_name] = class_   # keep track of the new class so that we can later create the generic bias folder


# create the general-purpose `ActivationRounder`
class WeightRounder(ComposedEditor):
    """Weight rounder."""
    def __init__(self):
        super(WeightRounder, self).__init__([class_() for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['WeightRounder']
