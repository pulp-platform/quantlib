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

from functools import reduce
from operator import mul
import torch
from typing import Dict, Tuple

from .statistics import TensorStatistic, NStatistic, MinStatistic, MaxStatistic, SumStatistic, Sum2Statistic
from quantlib.utils import quantlib_err_header

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


class TensorObserver(object):

    def __init__(self,
                 statistics: Dict[str, TensorStatistic],
                 subpopulation_dims: Tuple[int]):  # this is intended to be passed a `QGranularity` object, but in this way we keep it more general

        self._statistics = statistics

        # We used these parameters to ensure that the structure of the arrays
        # passed across different updates reflect a fixed "structure of
        # sub-populations".
        self._subpopulation_dims = subpopulation_dims
        self._ndim               = None
        self._permutation        = None
        self._broadcasting_shape = (1,) if len(self._subpopulation_dims) == 0 else None
        self._n_subpopulations   = None

    @property
    def broadcasting_shape(self) -> Tuple[int, ...]:
        return self._broadcasting_shape

    def _compute_broadcasting_shape(self, t: torch.Tensor) -> Tuple[int]:
        if len(self._subpopulation_dims) == 0:
            broadcasting_shape = (1,)
        else:
            broadcasting_shape = tuple(t.shape[i] if i in self._subpopulation_dims else 1 for i in range(0, t.ndim))
        return broadcasting_shape

    def _init_structural_attributes(self, t: torch.Tensor):

        if not set(self._subpopulation_dims).issubset(set(range(0, t.ndim))):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"expected Tensors with at least {max(self._subpopulation_dims)} dimensions, but received Tensor with {t.ndim} dimensions.")
        else:
            self._ndim               = t.ndim
            self._permutation        = tuple((*self._subpopulation_dims, *(i for i in range(0, t.ndim) if i not in self._subpopulation_dims)))
            self._broadcasting_shape = self._compute_broadcasting_shape(t)
            self._n_subpopulations   = 1 if len(self._subpopulation_dims) == 0 else reduce(mul, self._broadcasting_shape)

    @property
    def is_tracking(self) -> bool:
        return all(s.is_tracking for s in self._statistics.values())

    def _check_t(self, t: torch.Tensor) -> None:

        if not self.is_tracking:
            if t.ndim == 0:
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"only supports arrays with at least one dimension.")
            else:
                self._init_structural_attributes(t)

        else:
            if t.ndim != self._ndim:
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"expected an {self._ndim}-dimensional array, but received a {t.ndim}-dimensional one.")
            elif self._compute_broadcasting_shape(t) != self._broadcasting_shape:
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"expected an array with broadcasting shape {self._broadcasting_shape}, but received an array of shape {t.shape}.")
            else:
                pass

    def _make_broadcastable(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(self._broadcasting_shape)

    def update(self, t: torch.Tensor):

        self._check_t(t)
        t = t.permute(self._permutation)
        t = t.reshape(self._n_subpopulations, -1)

        for s in self._statistics.values():
            s.update(t)


class MinMaxMeanVarObserver(TensorObserver):

    def __init__(self, subpopulation_dims: Tuple[int]):
        super().__init__(
            {'n':    NStatistic(),
             'min':  MinStatistic(),
             'max':  MaxStatistic(),
             'sum':  SumStatistic(),
             'sum2': Sum2Statistic()},
            subpopulation_dims
        )

    @property
    def n(self) -> torch.Tensor:
        return self._make_broadcastable(self._statistics['n'].payload.values)

    @property
    def min(self) -> torch.Tensor:
        return self._make_broadcastable(self._statistics['min'].payload.values)

    @property
    def max(self) -> torch.Tensor:
        return self._make_broadcastable(self._statistics['max'].payload.values)

    @property
    def mean(self) -> torch.Tensor:
        device = self._statistics['sum'].payload.values.device
        return self._make_broadcastable(self._statistics['sum'].payload.values) / torch.as_tensor(self.n, device=device)

    @property
    def var(self) -> torch.Tensor:
        device = self._statistics['sum'].payload.values.device
        return self._make_broadcastable(self._statistics['sum2'].payload.values) / torch.as_tensor(self.n, device=device) - self.mean.pow(2)


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
