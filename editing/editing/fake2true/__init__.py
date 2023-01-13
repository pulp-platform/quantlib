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

from . import annotation
from . import epstunnels
from . import integerisation

#
# In the following, we define high-level `Editor`s (i.e., `ComposedEditor`s)
# to transform floating-point PyTorch networks into fake-quantised ones.
#
# Under the hood, `F2FConverter` breaks down into 2 `Annotator`s and 23 base
# base `Rewriter`s:
# * `ShapePropagator`;
# * `EpsPropagator`;
# * `EpsTunnelInserter`;
# * `EpsQLinearEpsIntegeriser`;
# * `EpsQConv1dEpsIntegeriser`;
# * `EpsQConv2dEpsIntegeriser`;
# * `EpsQConv3dEpsIntegeriser`;
# * `EpsQIdentityEpsRequantiser`;
# * `EpsQReLUEpsRequantiser`;
# * `EpsQReLU6EpsRequantiser`;
# * `EpsQLeakyReLUEpsRequantiser`;
# * `EpsBN1dQIdentityEpsRequantiser`;
# * `EpsBN1dQReLUEpsRequantiser`;
# * `EpsBN1dQReLU6EpsRequantiser`;
# * `EpsBN1dQLeakyReLUEpsRequantiser`;
# * `EpsBN2dQIdentityEpsRequantiser`;
# * `EpsBN2dQReLUEpsRequantiser`;
# * `EpsBN2dQReLU6EpsRequantiser`;
# * `EpsBN2dQLeakyReLUEpsRequantiser`;
# * `EpsBN3dQIdentityEpsRequantiser`;
# * `EpsBN3dQReLUEpsRequantiser`;
# * `EpsBN3dQReLU6EpsRequantiser`;
# * `EpsBN3dQLeakyReLUEpsRequantiser`;
# * `EpsTunnelSimplifier`;
# * `EpsTunnelRemover`.
#
# We also implement an example `F2T24bitConverter` showing how to inherit
# from the general-purpose `F2TConverter` to derive integerisation flows
# targetting a desired bit-shift value for requantisation operations.
#

from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from .annotation import InputDescription, InputDescriptionSpecType
from .annotation import F2TAnnotator
from .epstunnels import EpsTunnelInserter
from .integerisation import F2TIntegeriser
from .epstunnels import EpsTunnelConstructSimplifier
from .epstunnels import EpsTunnelRemover
from .epstunnels import FinalEpsTunnelRemover
from .canonicalisation import DropoutRemover
import torch.fx as fx
from quantlib.editing.editing.editors import ComposedEditor
from quantlib.editing.editing.editors.base.editor import Editor
from typing import Optional


class F2TConverter(ComposedEditor):

    def __init__(self,
                 B: int,
                 custom_editor: Optional[Editor] = None):

        editors_pre = [
            QuantLibRetracer(),
            F2TAnnotator(),
            EpsTunnelInserter(),
            F2TIntegeriser(B),
        ]

        editors_custom = [custom_editor] if custom_editor else []  # we assume that if `custom_editor` is defined, its `apply` method does not require extra arguments

        editors_post = [
            EpsTunnelConstructSimplifier(),
            EpsTunnelRemover(),
        ]

        super(F2TConverter, self).__init__(editors_pre + editors_custom + editors_post)

    def apply(self,
              g: fx.GraphModule,
              inputdescription: InputDescriptionSpecType = InputDescription(),
              *args,
              **kwargs) -> fx.GraphModule:

        g = self._children_editors[0](g)                    # `QuantLibRetracer`
        g = self._children_editors[1](g, inputdescription)  # `F2TAnnotator`
        for editor in self._children_editors[2:]:
            g = editor(g)

        return g


class F2T24bitConverter(F2TConverter):
    def __init__(self, custom_editor: Optional[Editor] = None):
        B: int = 24
        super(F2T24bitConverter, self).__init__(B, custom_editor)
