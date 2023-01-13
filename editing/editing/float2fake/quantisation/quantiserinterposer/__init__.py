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
import copy
import itertools
import torch.nn as nn

from ..qdescription import QDescriptionSpecType
from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from .finder import QuantiserInterposerMatcher
from .applier import QuantiserInterposerApplier
from quantlib.editing.editing.editors.nnmodules import NNModuleRewriter
from quantlib.editing.editing.editors import ComposedEditor

# set some constants to reduce code duplication
_N_FEATURES = 1
_KERNEL_SIZE = 1
_CONV_KWARGS = {'in_channels': _N_FEATURES, 'out_channels': _N_FEATURES, 'kernel_size': _KERNEL_SIZE}
_BN_KWARGS = {'num_features': _N_FEATURES}

# map roles to candidate `NNModuleDescription`s that could fit them
named_candidates = (
    ('Linear', NNModuleDescription(class_=nn.Linear,      kwargs={'in_features': 1, 'out_features': 1})),
    ('Conv1d', NNModuleDescription(class_=nn.Conv1d,      kwargs=_CONV_KWARGS)),
    ('Conv2d', NNModuleDescription(class_=nn.Conv2d,      kwargs=_CONV_KWARGS)),
    ('Conv3d', NNModuleDescription(class_=nn.Conv3d,      kwargs=_CONV_KWARGS)),
    ('BN1d',   NNModuleDescription(class_=nn.BatchNorm1d, kwargs=_BN_KWARGS)),
    ('BN2d',   NNModuleDescription(class_=nn.BatchNorm2d, kwargs=_BN_KWARGS)),
    ('BN3d',   NNModuleDescription(class_=nn.BatchNorm3d, kwargs=_BN_KWARGS)),
)

roles = Roles([
    ('linear_pre',  Candidates(copy.deepcopy(named_candidates))),
    ('linear_post', Candidates(copy.deepcopy(named_candidates))),
])

# define target patterns
problematic_successors = (
    ('Linear', ('Linear',)),
    ('Conv1d', ('Conv1d',)),
    ('Conv2d', ('Conv2d',)),
    ('Conv3d', ('Conv3d',)),
    ('BN1d',   ('Linear', 'Conv1d',)),
    ('BN2d',   ('Conv2d',)),
    ('BN3d',   ('Conv3d',))
)

admissible_screenplays = [itertools.product((pre,), post) for pre, post in problematic_successors]
admissible_screenplays = list(itertools.chain(*admissible_screenplays))


# programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern


def get_rewriter_class(class_name: str,
                       pattern:    NNSequentialPattern):

    def __init__(self_, qdescriptionspec):
        finder = QuantiserInterposerMatcher(pattern)
        applier = QuantiserInterposerApplier(qdescriptionspec, pattern)
        NNModuleRewriter.__init__(self_, class_name, pattern, finder, applier)

    class_ = type(class_name, (NNModuleRewriter,), {'__init__': __init__})

    return class_


namespace = OrderedDict([])
for name, pattern in generate_named_patterns(roles, admissible_screenplays):

    # create the new class
    class_name = name + 'QuantiserInterposer'

    class_ = get_rewriter_class(class_name, pattern)

    globals()[class_name] = class_   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    namespace[class_name] = class_   # keep track of the new class so that we can later create the generic quantiser interposer


# create the general-purpose `QuantiserInterposer`
class QuantiserInterposer(ComposedEditor):
    def __init__(self, qdescriptionspec: QDescriptionSpecType):
        super(QuantiserInterposer, self).__init__([class_(qdescriptionspec) for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['QuantiserInterposer']
