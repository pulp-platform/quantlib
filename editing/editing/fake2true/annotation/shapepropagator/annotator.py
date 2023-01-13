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

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
import warnings

from quantlib.editing.editing.editors import Annotator
from quantlib.editing.graphs.fx import FXOpcodeClasses, quantlib_symbolic_trace
from .shapeanddtype import InputShapesAndDTypes, InputShapesAndDTypesSpecType, resolve_inputshapesanddtypesspec
from quantlib.utils import quantlib_err_header, quantlib_wng_header


class ShapePropagator(Annotator):

    def __init__(self):
        name = 'ShapePropagator'
        super(ShapePropagator, self).__init__(name, quantlib_symbolic_trace)

    @staticmethod
    def is_shape_annotated(node: fx.Node) -> bool:
        return 'tensor_meta' in node.meta.keys()

    @staticmethod
    def clear_shape_annotations(g: fx.GraphModule) -> None:
        """Remove all shape annotations from the given graph."""
        for n in g.graph.nodes:
            try:
                del n.meta['tensor_meta']
            except KeyError:
                pass

    def apply(self,
              g: fx.GraphModule,
              shapesanddtypes: InputShapesAndDTypesSpecType = InputShapesAndDTypes()) -> fx.GraphModule:

        shapesanddtypes: InputShapesAndDTypes = resolve_inputshapesanddtypesspec(shapesanddtypes)

        # validate descriptions of input nodes
        inputs_graph     = [n.name for n in g.graph.nodes if (n.op in FXOpcodeClasses.PLACEHOLDER.value)]
        inputs_described = [n for n in shapesanddtypes.keys()]

        inputs_notdescribed = [n for n in inputs_graph if (n in set(inputs_graph).difference(set(inputs_described)))]
        inputs_unknown      = [n for n in inputs_graph if (n in set(inputs_described).difference(set(inputs_graph)))]

        if len(inputs_notdescribed) > 0:  # I do not have all the information I need
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"requires descriptions for all input nodes, but the following were not specified: {inputs_notdescribed}.")

        if len(inputs_unknown) > 0:  # I have all the information I need, but also spurious one
            warnings.warn(quantlib_wng_header(
                obj_name=self.__class__.__name__) + f"received descriptions for unknown placeholder nodes: {inputs_unknown}.")

        # propagate shapes
        # clear old shape annotations
        ShapePropagator.clear_shape_annotations(g)
        # create new shape annotations
        inputs = [torch.ones(shapesanddtypes[n].shape, dtype=shapesanddtypes[n].dtype) for n in inputs_graph]
        ShapeProp(g).run(*inputs)

        return g
