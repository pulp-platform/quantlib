import torch
import torch.nn as nn
from typing import Union

from quantlib.newalgorithms.qbase import QRangeSpecType, resolve_qrangespec, QRange
from quantlib.newalgorithms.qbase import QGranularitySpecType, resolve_qgranularityspec, QGranularity
from quantlib.newalgorithms.qbase import QHParamsInitStrategySpecType, resolve_qhparamsinitstrategyspec, QHParamsInitStrategy

from quantlib.newalgorithms.qbase import create_qhparams
from quantlib.newalgorithms.qbase import get_zero_scale, get_scale
from quantlib.newalgorithms.qbase import get_clipping_bounds
from quantlib.newalgorithms.qbase import MinMaxMeanVarObserver

from quantlib.newutils import UNKNOWN
from quantlib.newutils import quantlib_err_header


class _QModule(nn.Module):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 *args,
                 **kwargs):

        super().__init__()

        self._qrange: QRange                      = resolve_qrangespec(qrangespec)
        self._qgranularity: QGranularity          = resolve_qgranularityspec(qgranularityspec)
        self._qinitstrategy: QHParamsInitStrategy = resolve_qhparamsinitstrategyspec(qhparamsinitstrategyspec)
        self.register_buffer('_pin_offset', torch.Tensor([self._qrange.offset is not UNKNOWN]))
        self.register_buffer('_is_quantised', torch.Tensor([False]))

        self._observer: MinMaxMeanVarObserver = MinMaxMeanVarObserver(self._qgranularity)
        self.register_buffer('_is_observing', torch.Tensor([False]))

        self._qop: Union[torch.autograd.Function, None] = None
        self._register_qop(*args, **kwargs)  # this step is specific to the chosen PTQ/QAT algorithm

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

    def create_qhparams(self):
        raise NotImplementedError

    def init_qhparams(self):
        raise NotImplementedError

    def start_observing(self):
        raise NotImplementedError

    def stop_observing(self):
        raise NotImplementedError

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class _QActivation(_QModule):

    def __init__(self,
                 qrangespec: QRangeSpecType,
                 qgranularityspec: QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 *args,
                 **kwargs):

        super().__init__(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         *args,
                         **kwargs)

        if self._qgranularity != QGranularity(tuple()):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"does not support granularity other than per-array, but {self._qgranularity} was specified.")

    def create_qhparams(self):
        self._create_qhparams()

    def init_qhparams(self):
        self._init_qhparams()
        self._create_clipping_bounds()

    def start_observing(self):
        self._observer = MinMaxMeanVarObserver(self._qgranularity)  # reset observer by creating a new one
        self._is_observing |= True

    def stop_observing(self):
        self._is_observing &= False
        self.init_qhparams()
        self._observer = MinMaxMeanVarObserver(self._qgranularity)  # reset observer by creating a new one

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _QLinear(_QModule):

    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 *args,
                 **kwargs):

        super().__init__(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         *args,
                         **kwargs)

    def create_qhparams(self):
        self._observer = MinMaxMeanVarObserver(self._qgranularity)
        self._observer.update(self.weight)  # resolve broadcasting shape
        self._create_qhparams()
        self._observer = MinMaxMeanVarObserver(self._qgranularity)

    def init_qhparams(self):
        self._observer = MinMaxMeanVarObserver(self._qgranularity)
        self._observer.update(self.weight)  # resolve broadcasting shape
        self._init_qhparams()
        self._create_clipping_bounds()
        self._observer = MinMaxMeanVarObserver(self._qgranularity)

    def start_observing(self):
        self._is_observing |= True

    def stop_observing(self):
        self._is_observing &= False
        self.init_qhparams()

    def _register_qop(self, *args, **kwargs):
        raise NotImplementedError

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def qweight(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
