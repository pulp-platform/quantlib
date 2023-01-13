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

import collections
import torch
import torch.nn as nn
import torch.fx as fx

from typing import Union

from ...shapepropagator import ShapePropagator
from quantlib.editing.graphs.fx import unpack_then_split_fxnode_arguments
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule


UNDEFINED_EPS = torch.Tensor([float('nan')])


def is_eps_annotated(n: fx.Node) -> bool:
    return 'eps' in n.meta.keys()


ZERO_TOLERANCE    = 0.0
DEFAULT_TOLERANCE = 1e-8


def propagate_undefined(n: fx.Node,
                        m: Union[None, nn.Module],
                        *args,
                        **kwargs) -> torch.Tensor:
    """Return an undefined quantiser scale.

    The ``fx.Node``s using floating-point parameters (e.g., ``nn.Linear``,
    ``nn.ConvNd``, ``nn.BatchNormNd``) as well as those not enforcing
    fake-quantised tensors.
    """
    return UNDEFINED_EPS


def propagate_under_tolerance(n: fx.Node,
                              m: Union[None, nn.Module],
                              tolerance: float = DEFAULT_TOLERANCE,
                              *args,
                              **kwargs):
    """Compute the scales associated with the output of an ``fx.Node``.

    This function can be applied to single-input as well as to multi-input
    ``fx.Node``s. In the case of single-input ``fx.Node``s, the output scales
    will be equal to the input quantum; therefore, if the input is not
    fake-quantised (i.e., `eps_ins == [float('nan')]`), also the output will
    be marked as not fake-quantised (`eps_out = float('nan')`).

    The case of multi-input operations is meant to handle variadic operations
    such as additions and concatenations, where the semantic of the result
    (i.e., the quantum of the fake-quantised array) should coincide with the
    semantic of all the inputs to the operation. In these cases, the semantics
    of the inputs should already be coherent (i.e., all the quanta of the
    input fake-quantised arrays should coincide); we also allow for an
    approximated semantic equivalence by means of a non-negative tolerance
    parameter.

    In this last case, we verify that all the :math:`N` input quanta lie
    withing a given (either defined by the user or by the caller function)
    tolerance from each other. The computational cost of this verification
    is quadratic in :math:`N` (there are :math:`N * (N - 1) / 2` possible
    point pairs), but we assume that in most computational graphs :math:`N`
    will be sufficiently small not to notice this cost.
    """
    if tolerance < ZERO_TOLERANCE:
        raise ValueError("tolerance must be non-negative.")

    # unpack the `fx.Node`'s inputs
    fxnode_args, _, fxnode_kwargs, _ = unpack_then_split_fxnode_arguments(n)

    # find the inputs that are epsilon-annotated
    eps_annotated_kwargs = tuple(filter(lambda input_: is_eps_annotated(input_), fxnode_kwargs.values()))
    if len(eps_annotated_kwargs) > 0:
        raise RuntimeError("scale annotations must be limited to positional arguments.")

    eps_annotated_args = tuple(filter(lambda input_: is_eps_annotated(input_.fxnode), fxnode_args))
    eps_in = tuple(input_.fxnode.meta['eps'] for input_ in eps_annotated_args)

    # compute the scales of the output `torch.Tensor`
    if any((torch.any(e.isnan()) for e in eps_in)):  # I can't compute a numerical annotation
        eps_out = UNDEFINED_EPS

    else:
        # TODO: In the case of variadic inputs, we assume that all input
        #       arrays have the same shape, and anyway compatible granularity;
        #       we might want to support more precise comparisons and
        #       mixed-granularity in the future.
        flattened_eps_in = torch.vstack(tuple(map(lambda t: torch.flatten(t), eps_in)))
        min_eps = flattened_eps_in.amin(0)
        max_eps = flattened_eps_in.amax(0)
        diffs   = max_eps - min_eps  # should be equal to `torch.abs(max_eps - min_eps)` since scales are non-negative

        if torch.any(tolerance < diffs):  # I can not disambiguate the epsilon annotation
            eps_out = UNDEFINED_EPS

        else:  # we choose arbitrarily
            eps_out = eps_in[0]

    n.meta['eps'] = eps_out


def propagate_qmodules(n: fx.Node,
                       m: _QModule,
                       *args,
                       **kwargs):

    # unpack the `fx.Node`'s inputs
    fxnode_args, _, fxnode_kwargs, _ = unpack_then_split_fxnode_arguments(n)

    # find the inputs that are epsilon-annotated
    eps_annotated_kwargs = tuple(filter(lambda input_: is_eps_annotated(input_), fxnode_kwargs.values()))
    if len(eps_annotated_kwargs) > 0:
        raise RuntimeError("scale annotations must be limited to positional arguments.")

    eps_annotated_args = tuple(filter(lambda input_: is_eps_annotated(input_.fxnode), fxnode_args))
    eps_in = tuple(input_.fxnode.meta['eps'] for input_ in eps_annotated_args)

    try:
        eps_out = m.get_output_qhparams(eps_in)
    except AttributeError:
        raise AttributeError("[QuantLib] `nn.Module` {} does not support epsilon propagation.".format(type(m)))

    n.meta['eps'] = eps_out


def propagate_adaptiveavgpoolnd(n: fx.Node,
                                m: nn.Module,
                                *args,
                                **kwargs):

    if ShapePropagator.is_shape_annotated(n):

        # unpack the `fx.Node`'s inputs
        fxnode_args, _, fxnode_kwargs, _ = unpack_then_split_fxnode_arguments(n)

        # find the inputs that are epsilon-annotated
        eps_annotated_kwargs = tuple(filter(lambda input_: is_eps_annotated(input_), fxnode_kwargs.values()))
        if len(eps_annotated_kwargs) > 0:
            raise RuntimeError("scale annotations must be limited to positional arguments.")

        eps_annotated_args = tuple(filter(lambda input_: is_eps_annotated(input_.fxnode), fxnode_args))
        if len(eps_annotated_args) > 1:
            raise RuntimeError("AdaptiveAvgPoolNd expects a single argument.")

        p = eps_annotated_args[0].fxnode  # unique predecessor
        if ShapePropagator.is_shape_annotated(p):
            if n.meta['tensor_meta'].shape == p.meta['tensor_meta'].shape:
                eps_out = p.meta['eps']
            else:
                eps_out = UNDEFINED_EPS
        else:
            eps_out = UNDEFINED_EPS

    else:
        eps_out = UNDEFINED_EPS

    n.meta['eps'] = eps_out


EpsPropagationSpec = collections.namedtuple('EpsPropagationSpec', ['function', 'args', 'kwargs'])


_module_2_epspec = {
    nn.ReLU:      EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),  # TODO: I assume that zero is a valid quantisation level; e.g., the quantum of {-0.33, 0.17, 0.67} is 0.5, but the quantum of ReLU({-0.33, 0.17, 0.67}) = {0.0, 0.17, 0.67} is 0.01.
    nn.Identity:  EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    nn.Flatten:   EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    nn.MaxPool1d: EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    nn.MaxPool2d: EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    nn.MaxPool3d: EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    nn.AdaptiveAvgPool2d: EpsPropagationSpec(function=propagate_adaptiveavgpoolnd, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),  # TODO: This must be corrected: it just works if the effective spatial support has size 1x1.
    nn.Dropout: EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
}


_module_2_epspec.update({_QModule: EpsPropagationSpec(function=propagate_qmodules, args=[], kwargs={})})


_method_2_epspec = {
    'view':   EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    'add':    EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': DEFAULT_TOLERANCE}),
}


_function_2_epspec = {
    'flatten': EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': ZERO_TOLERANCE}),
    'add': EpsPropagationSpec(function=propagate_under_tolerance, args=[], kwargs={'tolerance': DEFAULT_TOLERANCE}),
}
