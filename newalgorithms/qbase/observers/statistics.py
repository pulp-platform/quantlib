from dataclasses import dataclass
import torch
from typing import Any

from quantlib.newutils import quantlib_err_header


@dataclass
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
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"expects two-dimensional arrays, but received an array of dimension {t.ndim}.")

        if self.is_tracking:
            n0 = t.shape[0]
            if n0 != self._payload.n0:
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"was tracking {self._payload.n0} sub-populations, but received {n0} samples.")

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
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "counter is overflowing!")

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
