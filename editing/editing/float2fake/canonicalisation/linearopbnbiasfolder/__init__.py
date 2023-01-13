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


from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from quantlib.editing.editing.editors.nnmodules import PathGraphMatcher
from .applier import LinearOpBNBiasApplier
from quantlib.editing.editing.editors.nnmodules import get_rewriter_class
from quantlib.editing.editing.editors import ComposedEditor


# set some constants to reduce code duplication
_N_FEATURES = 1
_KERNEL_SIZE = 1
_HAS_BIAS = True
_CONV_KWARGS = {'in_channels': _N_FEATURES, 'out_channels': _N_FEATURES, 'kernel_size': _KERNEL_SIZE, 'bias': _HAS_BIAS}
_BN_KWARGS = {'num_features': _N_FEATURES}
_LINEAROP_CHECKERS = (lambda m: m.bias is not None,)

# map roles to candidate `NNModuleDescription`s that could fit them
roles = Roles([

    ('linear', Candidates([
        ('Linear', NNModuleDescription(class_=nn.Linear, kwargs={'in_features': _N_FEATURES, 'out_features': _N_FEATURES, 'bias': _HAS_BIAS}, checkers=_LINEAROP_CHECKERS)),
        ('Conv1d', NNModuleDescription(class_=nn.Conv1d, kwargs=_CONV_KWARGS, checkers=_LINEAROP_CHECKERS)),
        ('Conv2d', NNModuleDescription(class_=nn.Conv2d, kwargs=_CONV_KWARGS, checkers=_LINEAROP_CHECKERS)),
        ('Conv3d', NNModuleDescription(class_=nn.Conv3d, kwargs=_CONV_KWARGS, checkers=_LINEAROP_CHECKERS)),
    ])),

    ('bn', Candidates([
        ('BN1d', NNModuleDescription(class_=nn.BatchNorm1d, kwargs=_BN_KWARGS)),
        ('BN2d', NNModuleDescription(class_=nn.BatchNorm2d, kwargs=_BN_KWARGS)),
        ('BN3d', NNModuleDescription(class_=nn.BatchNorm3d, kwargs=_BN_KWARGS)),
    ])),

])

# define target patterns
admissible_screenplays = [
    ('Linear', 'BN1d'),
    ('Conv1d', 'BN1d'),
    ('Conv2d', 'BN2d'),
    ('Conv3d', 'BN3d'),
]

# programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
namespace = OrderedDict([])
for name, pattern in generate_named_patterns(roles, admissible_screenplays):
    class_name = name + 'BiasFolder'
    class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, LinearOpBNBiasApplier)
    globals()[class_name] = class_   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    namespace[class_name] = class_   # keep track of the new class so that we can later create the generic bias folder


# create the general-purpose `LinearOpBNBiasFolder`
class LinearOpBNBiasFolder(ComposedEditor):
    """Generic bias folding utility."""
    def __init__(self):
        super(LinearOpBNBiasFolder, self).__init__([class_() for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['LinearOpBNBiasFolder']
