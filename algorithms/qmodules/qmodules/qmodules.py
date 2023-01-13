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

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Union

from quantlib.algorithms.qbase import QRangeSpecType, resolve_qrangespec, QRange
from quantlib.algorithms.qbase import QGranularitySpecType, resolve_qgranularityspec, QGranularity
from quantlib.algorithms.qbase import QHParamsInitStrategySpecType, resolve_qhparamsinitstrategyspec, QHParamsInitStrategy

from quantlib.algorithms.qbase import create_qhparams
from quantlib.algorithms.qbase import get_zero_scale, get_scale
from quantlib.algorithms.qbase import get_clipping_bounds
from quantlib.algorithms.qbase import MinMaxMeanVarObserver

from quantlib.utils import UNKNOWN
from quantlib.utils import quantlib_err_header


class _QModule(nn.Module):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
        """Register quantisation parameters into a target ``torch.nn.Module``.

        This class is meant to extend the functionalities of PyTorch
        ``Module``s with quantisation-related parameters. This constructor
        method is meant to be called after the constructor methods of the
        extended ``Module``s. In this way, several assumptions will hold:

        * ``self`` is an object of a class inheriting from ``nn.Module``,
          exposing methods such as ``register_buffer``;
        * if ``self`` is an object inheriting from linear ``nn.Module``s
          (e.g., ``nn.Linear`` or ``nn.Conv2d``), the ``weight`` attribute
          will be available.

        See the constructor methods of classes based on ``_QActivation`` and
        ``_QLinear`` to see how we tweak the MRO for the ``__init__`` methods
        to achieve this effect.

        As a consequence, we always assume that the constructor method of
        ``_QModule`` does not need to call the constructor methods of any
         class which preceeds it in the MRO of derived classes (i.e., derived
         from ``_QModule``).
        """

        self._qrange: QRange                      = resolve_qrangespec(qrangespec)
        self._qgranularity: QGranularity          = resolve_qgranularityspec(qgranularityspec)
        self._qinitstrategy: QHParamsInitStrategy = resolve_qhparamsinitstrategyspec(qhparamsinitstrategyspec)
        self.register_buffer('_pin_offset',   torch.tensor(self._qrange.offset is not UNKNOWN))
        self.register_buffer('_is_quantised', torch.tensor(False))

        self._observer: MinMaxMeanVarObserver = MinMaxMeanVarObserver(self._qgranularity)
        self.register_buffer('_is_observing', torch.tensor(False))

        self.create_qhparams()

        self._qop: Union[torch.autograd.Function, None] = None  # child classes should register an algorithm-specific `torch.autograd.Function`
        self._register_qop()

    def _create_qhparams(self):
        """Create quantiser hyper-parameters.

        This function registers ``torch.Tensor``s storing the values that
        describe the different quantisers associated with the slices
        determined by the ``Module``'s granularity.

        These buffers are not intended to be directly learnt. The ``n_levels``
        and ``step`` buffers are intended to be modified only by algorithms
        exploring mixed-precision. The ``zero`` and ``scale`` buffers are
        intended to be modified only by ``_QModule``'s children that implement
        quantisation algorithms that can learn quantisers (e.g., PACT or TQT).
        """
        zero, n_levels, step, scale = create_qhparams(self._qrange)
        self.register_buffer('zero',     torch.tile(zero,     self._observer.broadcasting_shape))
        self.register_buffer('n_levels', torch.tile(n_levels, self._observer.broadcasting_shape))
        self.register_buffer('step',     torch.tile(step,     self._observer.broadcasting_shape))
        self.register_buffer('scale',    torch.tile(scale,    self._observer.broadcasting_shape))

    def _init_qhparams(self):
        """Finalise the creation of quantiser hyper-parameters."""
        a, b = self._qinitstrategy.get_a_b(self._observer)
        if self._pin_offset:
            scale = get_scale(a, b, self.zero, self.n_levels, self.step)
            self.scale.data.copy_(scale.to(device=self.scale.device))
        else:
            zero, scale = get_zero_scale(a, b, self.n_levels, self.step)
            self.zero.data.copy_(zero.to(device=self.scale.device))
            self.scale.data.copy_(scale.to(device=self.scale.device))
        self._is_quantised |= True

    def _create_clipping_bounds(self):
        """Map quantiser hyper-parameters to clipping bounds.

        Quantisers are piece-wise constant, monotone, bounded functions. Due
        to these properties, the value of a quantiser must be constant outside
        a bounded subset of its domain (i.e., outside a given interval).
        Therefore, all the values lying outside the extremes :math:`\alpha`
        and :math:`\beta` of such an interval can be mapped back to the
        closest extreme before applying the function, without changing the
        result of the quantiser application.

        Several algorithms in the literature (e.g., PACT or TQT) learn such
        quantisation bounds to minimise the difference between the input
        values and their quantised images.
        """
        clip_lo, clip_hi = get_clipping_bounds(self.zero, self.n_levels, self.step, self.scale)
        self.register_parameter('clip_lo', nn.Parameter(clip_lo, requires_grad=False))
        self.register_parameter('clip_hi', nn.Parameter(clip_hi, requires_grad=False))

    def _set_clipping_bounds(self):
        clip_lo, clip_hi = get_clipping_bounds(self.zero, self.n_levels, self.step, self.scale)
        self.clip_lo.data = clip_lo
        self.clip_hi.data = clip_hi

    def create_qhparams(self):
        """Create ``nn.Module`` buffers hosting quantiser hyper-parameters."""
        raise NotImplementedError

    def init_qhparams(self):
        """Finalise ``nn.Module`` buffers hosting quantiser hyper-parameters."""
        raise NotImplementedError

    def start_observing(self):
        """Start collecting statistics about the unquantised array that the
        ``nn.Module`` is responsible for."""
        raise NotImplementedError

    def stop_observing(self):
        """Stop collecting statistics about the unquantised array that the
        ``nn.Module`` is responsible for; then, finalise buffers (i.e., fully
        specify the zero-point and scale parameters)."""
        raise NotImplementedError

    def _register_qop(self):
        # self._qop = [torch.autograd.Function_sub-class].apply
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Module,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> _QModule:
        """Special constructor to build ``_QModule``s from FP ``Module``s."""
        raise NotImplementedError

    def get_output_qhparams(self, in_scales: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class _QActivation(_QModule):

    def __init__(self,
                 qrangespec: QRangeSpecType,
                 qgranularityspec: QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        # TODO: ==============================================================
        # PyTorch is built around dynamic computational graphs. This design
        # choice implies that an activation `Module` does not have access to
        # shape information about the arrays it processes until a first
        # forward pass is run. Consequently, we can not know the broadcasting
        # shape for buffers storing quantiser hyper-parameters until the first
        # forward pass is run. We could create "placeholder" buffers to host
        # quantiser hyper-parameters such as `UninitalisedBuffer` (https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedBuffer.html),
        # buffers storing the `None` value, or buffers with shape `(1,)`.
        # However, in the general granularity case we would have to replace
        # the object they refer with a properly-shaped `torch.Tensor` when
        # the broadcasting shape information becomes available. However, if an
        # experiment crashes, we will have to build a new `_QActivation`
        # object from scratch when resuming the experiment, and this object
        # will again have an unknown shape. When `load_state_dict` notices a
        # mismatch between the shape of the new instantiated buffer object and
        # the shape of the buffer stored in memory, it will raise an error.
        # It seems that the PyTorch team does not intend to allow for this
        # behaviour:
        # * issue#8104 (https://github.com/pytorch/pytorch/issues/8104#issuecomment-395381167);
        # * issue#16675 (https://github.com/pytorch/pytorch/issues/16675#issuecomment-496113069);
        # * issue#24139 (https://github.com/pytorch/pytorch/issues/24139#issue-479210410).
        # Therefore, we just support per-array granularity when it comes to
        # activation `Module`s (luckily, this is also the most useful
        # practical use case).
        qgranularity = resolve_qgranularityspec(qgranularityspec)
        if qgranularity != QGranularity(tuple()):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"does not support granularity other than per-array, but {qgranularity} was specified.")
        # TODO: ==============================================================

        _QModule.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def create_qhparams(self):
        self._create_qhparams()
        self._create_clipping_bounds()

    def init_qhparams(self):
        self._init_qhparams()
        self._set_clipping_bounds()

    def start_observing(self):
        self._observer = MinMaxMeanVarObserver(self._qgranularity)  # reset observer by creating a new one
        self._is_observing |= True

    def stop_observing(self):
        self._is_observing &= False
        self.init_qhparams()
        self._observer = MinMaxMeanVarObserver(self._qgranularity)  # reset observer by creating a new one

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_observing:
            with torch.no_grad():
                self._observer.update(x)

        if self._is_quantised:
            x = self._call_qop(x)
        else:
            x = super(_QModule, self).forward(x)

        return x

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Module,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> _QActivation:
        """Special constructor to build ``_QActivation``s from FP ``Module``s."""
        raise NotImplementedError

    def get_output_qhparams(self, in_scales: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.scale.clone().detach()


class _QLinear(_QModule):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        _QModule.__init__(self,
                          qrangespec,
                          qgranularityspec,
                          qhparamsinitstrategyspec)

    def create_qhparams(self):
        self._observer = MinMaxMeanVarObserver(self._qgranularity)
        self._observer.update(self.weight)  # resolve broadcasting shape
        self._create_qhparams()
        self._create_clipping_bounds()
        self._observer = MinMaxMeanVarObserver(self._qgranularity)

    def init_qhparams(self):
        self._observer = MinMaxMeanVarObserver(self._qgranularity)
        self._observer.update(self.weight)  # resolve broadcasting shape
        self._init_qhparams()
        self._set_clipping_bounds()
        self._observer = MinMaxMeanVarObserver(self._qgranularity)

    def start_observing(self):
        self._is_observing |= True

    def stop_observing(self):
        self._is_observing &= False
        self.init_qhparams()

    def _register_qop(self):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def qweight(self):
        return self._call_qop(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # different linear `Module`s will call different functionals, to which weights should be explicitly passed

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Module,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> _QLinear:
        """Special constructor to build ``_QLinear``s from FP ``Module``s."""
        raise NotImplementedError

    def get_output_qhparams(self, in_scales: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        if len(in_scales) != 1:
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "expects exactly one fake-quantised input.")

        eps_in = in_scales[0]
        assert eps_in.numel() == 1

        if self._qgranularity == QGranularity(tuple()):
            out_scale = eps_in * self.scale.clone().detach()
        elif self._qgranularity == QGranularity((0,)):
            out_scale = eps_in * torch.swapaxes(self.scale.clone().detach(), 0, 1)  # the channel dimension is pushed to second place in feature arrays
        else:
            raise NotImplementedError

        return out_scale
