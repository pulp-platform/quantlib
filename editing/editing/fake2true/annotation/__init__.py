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

from .shapepropagator import *
from .epspropagator import *


import torch.fx as fx
from quantlib.editing.editing.editors import ComposedEditor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from .inputdescription import PlaceholderDescription, InputDescription, InputDescriptionSpecType, resolve_inputdescriptionspec


class F2TAnnotator(ComposedEditor):

    def __init__(self):
        super(F2TAnnotator, self).__init__([
            QuantLibRetracer(),
            ShapePropagator(),
            EpsPropagator(),
        ])

    def apply(self,
              g: fx.GraphModule,
              inputdescriptionspec: InputDescriptionSpecType = InputDescription(),
              *args,
              **kwargs) -> fx.GraphModule:

        # unpack input descriptions into shape/type annotations and scale annotations
        inputdescription = resolve_inputdescriptionspec(inputdescriptionspec)

        shapesanddtypes = InputShapesAndDTypes()
        inputscales     = InputScales()
        for target, (shape, dtype, scale) in inputdescription.items():
            shapesanddtypes[target] = ShapeAndDType(shape=shape, dtype=dtype)
            inputscales[target] = scale

        # annotate
        g = self._children_editors[0](g)
        g = self._children_editors[1](g, shapesanddtypes)
        g = self._children_editors[2](g, inputscales)

        return g
