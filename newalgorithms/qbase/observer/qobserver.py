import enum
import torch
from typing import Tuple, Union

from ..granularity import QGranularity


@enum.unique
class QUnquantisedBounds(enum.Enum):
    CONST = 0
    MINMAX = 1
    DIST = 3


class QUnquantisedObserver(object):

    def __init__(self,
                 granularity: QGranularity,
                 t: torch.Tensor):
        """An object to track the statistics of a collection of arrays.

        Quantisation-aware training (QAT) algorithms can yield better
        performance if the quantisation parameters of the target networks are
        initialised according to the statistics of the floating-point data
        that passes through the unquantised version.

        This object tracks elementary statistics that can be used downstream
        to initialise the quantisation parameters of fake-quantised
        ``nn.Module``s:

        * the maximum component value encountered;
        * the minimum component value encountered;
        * a smoothed average of the components' mean, with interpolation
          factor 0.1;
        * a smoothed average of the components' variance, with interpolation
          factor 0.1.

        Arguments:
            granularity: an enumerated describing whether to track statistics
                about the whole :math:`N`-dimensional arrays or about the
                individual :math:`N-1`-dimensional slices identified along the
                first dimension when :math:`N \geq 2`. The first case is
                equivalent to supposing that the array components are IID
                samples from a single distribution; the second case is
                the second case amounts to considering :math:`N_{0}` different
                distributions, each of which is associated with a slice of the
                input array.
            t: a ``Tensor`` which carries information about the data type and
                the device of the ``Tensor``s that are to be tracked (i.e.,
                the target ``Tensor``s. This information ensures that the
                statistics-tracking ``Tensor``s are homogeneous to the target
                ``Tensor``s, and therefore no issue should be raised due to
                runtime incompatibilites. Since the target ``Tensor``s are
                usually combined with ``nn.Parameter``s of the hosting
                ``nn.Module``, ``nn.Parameter``s should be compatible, and I
                suggest to pass such an object.

        """

        self._granularity = granularity

        # preserve compatibility of operations with `torch.Tensor`s
        self._dtype = None
        self._device = None

        # the number of distributions to track
        self._n0 = None

        # helper statistics
        self._n = None
        self._sum = None
        self._sum2 = None

        # target statistics
        self._min = None
        self._max = None

        self.reset(t)

    @property
    def _mean(self) -> torch.Tensor:
        return self._sum / self._n

    @property
    def _var(self) -> torch.Tensor:
        return self._sum2 / self._n - self._mean.pow(2)

    def reset(self, t: Union[torch.Tensor, None] = None):

        if t is not None:
            self._dtype = t.dtype
            self._device = t.device

        self._n0 = None

        self._n = torch.Tensor([0]).to(dtype=torch.int64, device=self._device)
        self._sum = torch.Tensor([0.0]).to(dtype=self._dtype, device=self._device)
        self._sum2 = torch.Tensor([0.0]).to(dtype=self._dtype, device=self._device)
        self._min = torch.Tensor([float('inf')]).to(dtype=self._dtype, device=self._device)
        self._max = torch.Tensor([-float('inf')]).to(dtype=self._dtype, device=self._device)

    def _check_n_overflow(self, n: int):
        """Check that the sample counter is not overflowing!"""
        if (self._n + n) - n != self._n:
            raise RuntimeError(f"[QuantLab] {self.__class__.__name__} counters are overflowing!")

    def update(self, x: torch.Tensor) -> None:
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

        if self._granularity is QGranularity.PER_LAYER:

            # update counter
            self._check_n_overflow(x.numel())
            self._n += x.numel()

            # update partial sums
            self._sum = self._sum + x.sum()
            self._sum2 = self._sum2 + x.pow(2).sum()

            # update min/max
            self._min = torch.min(self._min, x.amin())
            self._max = torch.max(self._max, x.amax())

        else:  # self._granularity is QGranularity.PER_CHANNEL:

            # the number of tracked distributions is determined by the first array passed after a call to `reset`
            if self._n0 is None:
                self._n0 = x.shape[0]
            else:
                if x.shape[0] != self._n0:
                    raise ValueError(f"[QuantLab] {self.__class__.__name__} was tracking {self._n0} distributions, but received samples from {x.shape[0]}.")

            # can the array represent a collection of samples from multiple distributions?
            if x.ndim == 0:
                raise ValueError(f"[QuantLab] {self.__class__.__name__} can track per-channel statistics only for arrays containing multiple components.")
            else:
                x = x.view(self._n0, -1)  # make the array 2-dimensional

            # update counter
            self._check_n_overflow(int(x.numel() / self._n0))
            self._n += int(x.numel() / self._n0)

            # update partial sums
            self._sum = self._sum + x.sum(dim=tuple(range(1, x.ndim)))
            self._sum2 = self._sum2 + x.pow(2).sum(dim=tuple(range(1, x.ndim)))

            # update min/max
            self._min = torch.min(self._min, x.amin(dim=tuple(range(1, x.ndim))))
            self._max = torch.max(self._max, x.amax(dim=tuple(range(1, x.ndim))))

    def get_bounds(self,
                   strategy: QUnquantisedBounds) -> Tuple[torch.Tensor, torch.Tensor]:

        if strategy is QUnquantisedBounds.CONST:
            clip_lo = torch.ones_like(self._min) * -1.0
            clip_hi = torch.ones_like(self._max) * 1.0

        elif strategy is QUnquantisedBounds.MINMAX:
            clip_lo = self._min
            clip_hi = self._max

        elif strategy is QUnquantisedBounds.DIST:
            # I borrow the "golden rule" of Gaussian distributions, where
            # 99.7% of the probability mass lies within three standard
            # deviations from the mean.
            n_std = 3
            clip_lo = self._mean - n_std * self._var.sqrt()
            clip_hi = self._mean + n_std * self._var.sqrt()

        else:
            raise ValueError(f"[QuantLab] QStatisticsTracker can not return distribution bounds with strategy {strategy}.")

        return clip_lo, clip_hi
