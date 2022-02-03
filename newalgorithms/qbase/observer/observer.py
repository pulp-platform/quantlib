import dataclasses
from functools import reduce
from operator import mul
import torch
from typing import Any, List, Tuple, Optional

from quantlib.newutils import quantlib_err_header

# initialised from statistics
#                                  * pinned     * int offset     * float offset
# qrange any
# qrange unsigned
# qrange signed (symmetric)
# qrange signed (quasisymmetric)
# qrange sign (special)

# initialised by the user
#                                  * pinned     * int offset     * float offset
# qrange any
# qrange unsigned
# qrange signed (symmetric)
# qrange signed (quasisymmetric)
# qrange sign (special)


@dataclasses.dataclass
class StatisticPayload:
    n0: int      # number of sub-populations
    values: Any  # running values of the observed statistic for all the sub-populations


class TensorStatistic(object):
    """A class to track running statistics of different sub-populations.

    Arrays can be interpreted as samples from unknown probability
    distributions. In particular, two-dimensional arrays with shape
    :math:`N_{0} \times N_{1}` can be interpreted as collections of
    :math:`N_{1}`-samples from :math:`N_{0}` different distributions. Due to
    the homogeneity of row sizes, each sample contains the same number of data
    points as its siblings.

    When we need to track the statistics in a running fashion (i.e., when
    samples from the :math:`N_{0}` observed distributions are provided
    sequentially), the number :math:`N_{0}` of samples at each iteration must
    be constant.

    """

    def __init__(self):
        self._payload = None

    @property
    def payload(self) -> StatisticPayload:
        return self._payload

    @property
    def is_tracking(self) -> bool:
        return self._payload is not None

    def _check_t(self, t: torch.Tensor) -> None:

        if t.ndim != 2:
            raise ValueError(quantlib_err_header(self.__class__.__name__) + f"expects two-dimensional arrays, but received an array of dimension {t.ndim}.")

        if self.is_tracking:
            n0 = t.shape[0]
            if n0 != self._payload.n0:
                raise ValueError(quantlib_err_header(self.__class__.__name__) + f"was tracking {self._payload.n0} sub-populations, but received {n0} samples.")

    def _update(self, t: torch.Tensor) -> None:
        """Update the running value of the statistic."""
        raise NotImplementedError

    def update(self, t: torch.Tensor) -> None:
        self._check_t(t)
        self._update(t)


class NStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _check_n_overflow(self, n: int):
        """Check that the sample counter is not overflowing!"""
        if (self._payload.values + n) - self._payload.values != n:
            raise RuntimeError(quantlib_err_header(self.__class__.__name__) + "counter is overflowing!")

    def _update(self, t: torch.Tensor):

        n = t.shape[-1]

        if not self.is_tracking:
            n0 = t.shape[0]
            self._payload = StatisticPayload(n0, n)
        else:
            self._check_n_overflow(n)
            self._payload.values = self._payload.values + n


class MinStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        min_ = torch.amin(t, dim=-1)

        if not self.is_tracking:
            n0 = t.shape[0]
            self._payload = StatisticPayload(n0, min_)
        else:
            self._payload.values = torch.min(self._payload.values, min_)


class MaxStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        max_ = torch.amax(t, dim=-1)

        if not self.is_tracking:
            n0 = t.shape[0]
            self._payload = StatisticPayload(n0, max_)
        else:
            self._payload.values = torch.max(self._payload.values, max_)


class SumStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        sum_ = torch.sum(t, dim=-1)

        if not self.is_tracking:
            n0 = t.shape[0]
            self._payload = StatisticPayload(n0, sum_)
        else:
            self._payload.values = self._payload.values + sum_


class Sum2Statistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        sum2 = torch.sum(t.pow(2), dim=-1)

        if not self.is_tracking:
            n0 = t.shape[0]
            self._payload = StatisticPayload(n0, sum2)
        else:
            self._payload.values = self._payload.values + sum2


class TensorObserver(object):

    def __init__(self,
                 statistics: List[TensorStatistic],
                 subpopulation_dims: Optional[Tuple[int, ...]] = None):

        self._statistics = statistics
        self._subpopulation_dims = subpopulation_dims
        self._ndim = None

    @property
    def is_tracking(self) -> bool:
        return all(s.is_tracking for s in self._statistics)

    def _check_t(self, t: torch.Tensor) -> None:
        if self._subpopulation_dims is not None:
            if not set(set(self._subpopulation_dims)).issubset(range(0, t.ndim)):
                raise ValueError(quantlib_err_header(self.__class__.__name__) + f"expected an array with at least {max(self._subpopulation_dims)} dimensions, but got an array with dimension {t.ndim}.")

    def _rearrange_tensor(self, t: torch.Tensor) -> torch.Tensor:

        #              sub_dims == None       sub_dims != None
        # t.ndim == 0  t.reshape(1, -1)       t.reshape(1, 1)
        # t.ndim == 1  t.reshape(1, -1)       ambiguous...
        # t.ndim > 1   t.reshape(1, -1)       sub_dims \subset range(0, t.ndim), then rearrange

        if self._subpopulation_dims is None:
            t = t.reshape(1, -1)

        else:
            if t.ndim == 0:
                t = t.reshape(1, 1)
            elif t.ndim == 1:
                raise ValueError(quantlib_err_header(self.__class__.__name__) + f"can not disambiguate `torch.Tensor` {t}: does it represent {t.numel()} one-samples from {t.numel()} populations or one {t.numel()}-sample from one population?")
            else:
                # bring sub-population dimensions in front
                dims_permutation = (*self._subpopulation_dims, *tuple(i for i in range(0, t.ndim) if i not in self._subpopulation_dims))
                t = t.permute(dims_permutation)
                # rearrange the array into a two-dimensional array (rows are samples, columns are sample components)
                n_subpopulations = reduce(mul, (t.shape[i] for i in self._subpopulation_dims))
                t = t.permute(dims_permutation).reshape(n_subpopulations, -1)

        return t

    def update(self, t: torch.Tensor):

        self._check_t(t)
        t = self._rearrange_tensor(t)

        for s in self._statistics:
            s.update(t)


class MinMaxMeanVarObserver(TensorObserver):

    def __init__(self, subpopulation_dims: Optional[Tuple[int, ...]] = None):
        super().__init__(
            [NStatistic(),
             MinStatistic(),
             MaxStatistic(),
             SumStatistic(),
             Sum2Statistic()],
            subpopulation_dims
        )

    @property
    def n(self) -> int:
        return self._statistics[0].payload.values

    @property
    def min(self) -> torch.Tensor:
        return self._statistics[1].payload.values

    @property
    def max(self) -> torch.Tensor:
        return self._statistics[2].payload.values

    @property
    def mean(self) -> torch.Tensor:
        return self._statistics[3].payload.values / self.n

    @property
    def var(self) -> torch.Tensor:
        return self._statistics[4].payload.values / self.n - self.mean.pow(2)


"""Track the statistics of a collection of arrays.

Quantisation-aware training (QAT) algorithms can yield better
performance if the quantisation parameters of the target networks are
initialised according to the statistics of the floating-point data
that passes through the unquantised version.

This object tracks elementary statistics that can be used downstream
to initialise the quantisation parameters of fake-quantised
``nn.Module``s:

"""

"""Update the running values of the tracked statistics.

This object is not in charge of ensuring that the shapes of the arrays
passed to it are consistent through multiple iterations. I only assume
that zero- and one-dimensional arrays represent samples from a single
distribution (i.e., they are not compatible with per-slice tracking),
whereas two-or-more-dimensional arrays can be considered samples from
:math:`N_{0}` different distributions. In this last case, the objects
expects samples from the same number of distributions at each call, so
this quantity must not change in-between iterations.

Arguments:
    x: the array whose components are to be tracked.

"""

# import enum
#
# from ..qspecs import QGranularity
#
#
# @enum.unique
# class QUnquantisedBounds(enum.Enum):
#     CONST = 0
#     MINMAX = 1
#     DIST = 3
#
# if strategy is QUnquantisedBounds.CONST:
#     clip_lo = torch.ones_like(self._min) * -1.0
#     clip_hi = torch.ones_like(self._max) * 1.0
#
# elif strategy is QUnquantisedBounds.MINMAX:
#     clip_lo = self._min
#     clip_hi = self._max
#
# elif strategy is QUnquantisedBounds.DIST:
#     # I borrow the "golden rule" of Gaussian distributions, where
#     # 99.7% of the probability mass lies within three standard
#     # deviations from the mean.
#     n_std = 3
#     clip_lo = self._mean - n_std * self._var.sqrt()
#     clip_hi = self._mean + n_std * self._var.sqrt()
#
# else:
#     raise ValueError(f"[QuantLab] QStatisticsTracker can not return distribution bounds with strategy {strategy}.")
