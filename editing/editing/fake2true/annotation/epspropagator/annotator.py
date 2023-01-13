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

import copy
import torch
import torch.fx as fx
import warnings

from .propagationrules import UNDEFINED_EPS, is_eps_annotated
from .propagationrules import _module_2_epspec
from .propagationrules import _method_2_epspec
from .propagationrules import _function_2_epspec

from quantlib.editing.editing.editors import Annotator
from quantlib.editing.graphs.fx import FXOpcodeClasses, quantlib_symbolic_trace
from .inputscales import InputScales, InputScalesSpecType, resolve_inputscalesspec
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.utils import quantlib_wng_header


class EpsPropagator(Annotator):
    """A class to annotate ``fx.GraphModule``s with quantiser scales."""

    def __init__(self):
        name = 'EpsPropagator'
        super(EpsPropagator, self).__init__(name, quantlib_symbolic_trace)

    @staticmethod
    def returns_qtensor(n: fx.Node):
        return is_eps_annotated(n) and not torch.any(n.meta['eps'].isnan())

    @staticmethod
    def clear_eps_annotations(g: fx.GraphModule):
        """Remove all scale annotations from the given graph."""
        for n in g.graph.nodes:
            try:
                del n.meta['eps']
            except KeyError:
                pass

    def apply(self,
              g: fx.GraphModule,
              inputscales: InputScalesSpecType = InputScales()) -> fx.GraphModule:

        inputscales = resolve_inputscalesspec(inputscales)

        # validate descriptions of input nodes
        inputs_graph     = [n.target for n in g.graph.nodes if (n.op in FXOpcodeClasses.PLACEHOLDER.value)]
        inputs_described = [n for n in inputscales.keys()]

        inputs_notdescribed = [n for n in inputs_graph if (n in set(inputs_graph).difference(set(inputs_described)))]
        inputs_unknown      = [n for n in inputs_graph if (n in set(inputs_described).difference(set(inputs_graph)))]

        if len(inputs_notdescribed) > 0:  # canonicalise: set the scale of unspecified nodes to "unknown"
            warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + f"did not receive scale information for the following nodes: {inputs_unknown}.")
            for n in inputs_notdescribed:
                inputscales[n] = UNDEFINED_EPS

        if len(inputs_unknown) > 0:  # I have all the information I need, but also spurious one
            warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + f"received descriptions for unknown placeholder nodes: {inputs_unknown}.")

        # propagate scales
        # clear old scale annotations
        EpsPropagator.clear_eps_annotations(g)

        # create new scale annotations
        for n in g.graph.nodes:
            # I assume that the nodes in the `fx.Graph` are topologically sorted.
            # Although the documentation of torch.fx is a bit opaque about this property,
            # we observe that if they were not topologically sorted then the IR might be
            # invalid. In fact, imagine that an `fx.Node` in position :math:`i` were
            # taking in input at least an `fx.Node` in position :math:`j > i`. This would
            # mean that the IR is suggesting that is possible to execute node :math:`i`
            # before node :math:`j`, violating the dependency constraint.

            if n.op in FXOpcodeClasses.PLACEHOLDER.value:
                n.meta['eps'] = inputscales[n.target]  # annotate by looking-up the value

            elif n.op in FXOpcodeClasses.OUTPUT.value:
                if not ((len(n.args) == 1) and (len(n.kwargs) == 0)):
                    raise RuntimeError  # output nodes should copy the output of a single operation
                n.meta['eps'] = next(iter(n.args)).meta['eps']  # annotate by copying the (unique) input's scale

            elif n.op in FXOpcodeClasses.CALL_MODULE.value:
                m = g.get_submodule(target=n.target)
                try:
                    class_ = type(m) if not isinstance(m, _QModule) else _QModule
                    epspec = _module_2_epspec[class_]
                    epspec.function(n, m, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    # I assume that each `call_module` `fx.Node` yields a `torch.Tensor` which has a
                    # valid semantic with respect to the functionality that the network is designed
                    # to solve (e.g., a feature map). Therefore, if the epsilon propagation rule for
                    # a given `call_module` node is not defined, I return the "undefined" value ('NaN').
                    n.meta['eps'] = UNDEFINED_EPS

            elif n.op in FXOpcodeClasses.CALL_METHOD.value:
                try:
                    epspec = _method_2_epspec[n.target]
                    epspec.function(n, None, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    # Differently from `call_module` `fx.Node`s, there are `call_method` nodes that
                    # yield values which do not have a valid semantic with respect to the functionality
                    # that the network is designed to achieve (e.g., evaluating the `size` of a given
                    # `torch.Tensor`). Therefore, the default behaviour here is skipping the annotation.
                    continue  # TODO

            elif n.op in FXOpcodeClasses.CALL_FUNCTION.value:
                try:
                    epspec = _function_2_epspec[n.target.__name__]
                    epspec.function(n, None, *copy.copy(epspec.args), **copy.copy(epspec.kwargs))
                except KeyError:
                    continue  # TODO

            else:  # `n.op in FXOpcodeClasses.GET_ATTR.value
                continue  # TODO

        return g
