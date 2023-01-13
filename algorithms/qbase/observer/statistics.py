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

from dataclasses import dataclass
import torch
from typing import Any

from quantlib.utils import quantlib_err_header


@dataclass
class StatisticPayload:
    n_subpopulations: int  # number of sub-populations
    values: Any            # running values of the observed statistic for all the sub-populations


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
            n_subpopulations = t.shape[0]
            if n_subpopulations != self._payload.n_subpopulations:
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"was tracking {self._payload.n_subpopulations} sub-populations, but received {n_subpopulations} samples.")

    def _update(self, t: torch.Tensor) -> None:
        """Update the running value of the statistic."""
        raise NotImplementedError

    def update(self, t: torch.Tensor) -> None:
        self._check_t(t)
        self._update(t)


class NStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _check_n_overflow(self, n: torch.Tensor):
        """Check that the sample counter is not overflowing!"""
        if (self._payload.values + n) - self._payload.values != n:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "counter is overflowing!")

    def _update(self, t: torch.Tensor):

        n = t.shape[-1]

        if not self.is_tracking:
            n_subpopulations = t.shape[0]
            n = torch.ones(n_subpopulations) * n
            self._payload = StatisticPayload(n_subpopulations, n)
        else:
            n = torch.ones(self._payload.n_subpopulations) * n
            self._check_n_overflow(n)
            self._payload.values = self._payload.values + n


class MinStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        min_ = torch.amin(t, dim=-1)

        if not self.is_tracking:
            n_subpopulations = t.shape[0]
            self._payload = StatisticPayload(n_subpopulations, min_)
        else:
            self._payload.values = torch.min(self._payload.values, min_)


class MaxStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        max_ = torch.amax(t, dim=-1)

        if not self.is_tracking:
            n_subpopulations = t.shape[0]
            self._payload = StatisticPayload(n_subpopulations, max_)
        else:
            self._payload.values = torch.max(self._payload.values, max_)


class SumStatistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        sum_ = torch.sum(t, dim=-1)

        if not self.is_tracking:
            n_subpopulations = t.shape[0]
            self._payload = StatisticPayload(n_subpopulations, sum_)
        else:
            self._payload.values = self._payload.values + sum_


class Sum2Statistic(TensorStatistic):

    def __init__(self):
        super().__init__()

    def _update(self, t: torch.Tensor):

        sum2 = torch.sum(t.pow(2), dim=-1)

        if not self.is_tracking:
            n_subpopulations = t.shape[0]
            self._payload = StatisticPayload(n_subpopulations, sum2)
        else:
            self._payload.values = self._payload.values + sum2
