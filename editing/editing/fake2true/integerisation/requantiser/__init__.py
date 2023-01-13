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

from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from .finder import RequantiserMatcher
from .applier import RequantiserApplier
from quantlib.editing.editing.editors.nnmodules import NNModuleRewriter
from quantlib.editing.editing.editors import ComposedEditor
from quantlib.editing.graphs.nn import EpsTunnel


# set some constants to reduce code duplication
_BN_KWARGS = {'num_features': 1}
_EPS_KWARGS = {'eps': torch.Tensor([1.0])}

# map roles to candidate `NNModuleDescription`s that could fit them
roles = Roles([

    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),

    ('bn', Candidates([
        ('',     None),  # batch-normalisation is an optional part of the pattern
        ('BN1d', NNModuleDescription(class_=nn.BatchNorm1d, kwargs=_BN_KWARGS)),
        ('BN2d', NNModuleDescription(class_=nn.BatchNorm2d, kwargs=_BN_KWARGS)),
        ('BN3d', NNModuleDescription(class_=nn.BatchNorm3d, kwargs=_BN_KWARGS)),
    ])),

    ('activation', Candidates([
        ('QIdentity',  NNModuleDescription(class_=nn.Identity,  kwargs={})),
        ('QReLU',      NNModuleDescription(class_=nn.ReLU,      kwargs={})),
        ('QReLU6',     NNModuleDescription(class_=nn.ReLU6,     kwargs={})),
        ('QLeakyReLU', NNModuleDescription(class_=nn.LeakyReLU, kwargs={})),
    ])),

    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),
])

# define target patterns
admissible_screenplays = list(roles.all_screenplays)


# programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern


def get_rewriter_class(class_name: str,
                       pattern: NNSequentialPattern):
    def __init__(self_, B):
        finder = RequantiserMatcher(pattern)
        applier = RequantiserApplier(pattern, B)
        NNModuleRewriter.__init__(self_, class_name, pattern, finder, applier)

    class_ = type(class_name, (NNModuleRewriter,), {'__init__': __init__})

    return class_


namespace = OrderedDict([])
for name, pattern in generate_named_patterns(roles, admissible_screenplays):
    class_name = name + 'Requantiser'

    class_ = get_rewriter_class(class_name, pattern)

    globals()[class_name] = class_   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    namespace[class_name] = class_   # keep track of the new class so that we can later create the generic requantiser


# create the general-purpose `Requantiser`
class Requantiser(ComposedEditor):
    def __init__(self, B: int):
        super(Requantiser, self).__init__([class_(B) for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['Requantiser']
