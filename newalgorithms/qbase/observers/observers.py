from functools import reduce
from operator import mul
import torch
from typing import List, Tuple, Optional

from .statistics import TensorStatistic, NStatistic, MinStatistic, MaxStatistic, SumStatistic, Sum2Statistic
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
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"expected an array with at least {max(self._subpopulation_dims)} dimensions, but got an array with dimension {t.ndim}.")

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
                raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"can not disambiguate `torch.Tensor` {t}: does it represent {t.numel()} one-samples from {t.numel()} populations or one {t.numel()}-sample from one population?")
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
