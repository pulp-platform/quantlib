"""The archetypes of the QNN layers that can be trained using QAT algorithms.

In QuantLib, we support the training of feedforward neural networks with
quantisation-aware training (QAT) algorithms.

We support fully-feedforward networks (i.e., networks where for any given pair
of parameter arrays or feature maps there is a single processing path
connecting them, yielding purely sequential computational graphs), as well as
feedforward networks where this constraint can be softened (e.g., networks of
the ResNet family).

We support training of both linear operations and activation operations (both
non-linear ones and the identity).

"""


import dataclasses
import torch
import torch.nn as nn
from typing import Union


@dataclasses.dataclass
class QStatistics:
    min: torch.Tensor
    max: torch.Tensor
    mean: torch.Tensor
    var: torch.Tensor


class QATActivation(nn.Module):
    """A base class archetyping QAT-trainable activation operations."""

    def __init__(self,
                 quantised: bool):

        super().__init__()

        self._is_quantising = quantised
        self._granularity = QGranularity['PER_LAYER']  # we do not support per-channel granularity in feature maps

        self._is_tracking = False
        self._qstatistics = None

    @property
    def is_tracking(self) -> bool:
        return self._is_tracking

    def _init_tracking(self, t: torch.Tensor):
        """Initialise the structures that will store the statistics.

        Arguments:
            t: a ``Tensor`` which carries information about the data type and
                the device of the ``Tensor``s that are to be tracked (i.e.,
                the target ``Tensor``s. This information ensures that the
                statistics-tracking ``Tensor``s are homogeneous to the target
                ``Tensor``s, and therefore no issue should be raised due to
                runtime incompatibilites. Since the target ``Tensor``s are
                usually combined with ``nn.Parameter``s of the hosting
                ``nn.Module``, ``nn.Parameter``s shoudl be compatibel, and I
                suggest to pass such an object.

        """
        self._qstatistics = QStatistics(torch.Tensor([float('inf')]).to(dtype=t.dtype, device=t.device),
                                        torch.Tensor([-float('inf')]).to(dtype=t.dtype, device=t.device),
                                        torch.Tensor([0.0]).to(dtype=t.dtype, device=t.device),
                                        torch.Tensor([1.0]).to(dtype=t.dtype, device=t.device))

    def _track_statistics(self, x: torch.Tensor) -> None:
        """Collect statistics about the input array.

        Quantisation-aware training (QAT) algorithms can yield better
        performance if the quantisation parameters of the target networks are
        initialised according to the statistics of the floating-point data
        that passes through the unquantised version.

        This function computes some elementary statistics that can be used
        downstream to initialise the quantisation parameters of the
        ``nn.Module``:

        * the maximum component value encountered;
        * the minimum component value encountered;
        * a smoothed average of the components' mean, with interpolation
          factor 0.1;
        * a smoothed average of the components' variance, with interpolation
          factor 0.1.

        """

        with torch.no_grad():
            self._qstatistics.min = torch.min(self._qstatistics.min, x.min())
            self._qstatistics.max = torch.max(self._qstatistics.max, x.max())
            self._qstatistics.mean = 0.9 * self._qstatistics.mean + 0.1 * x.mean()
            self._qstatistics.var = 0.9 * self._qstatistics.var + 0.1 * (x.var() ** 2)

    def start_tracking(self) -> None:
        self._is_tracking = True
        self._init_tracking()

    def stop_tracking(self) -> None:
        self._is_tracking = False
        self._qstatistics = None

    def init_clipping_bounds(self,
                             statistics_aware: bool = False,
                             statistics_aware_strategy: Union[None, str] = None):

        if statistics_aware:
            if self._qstatistics is None:
                raise ValueError("[QuantLab] QActivation can't initialise clipping bounds from statistics without statistics!")
            else:
                if statistics_aware_strategy is None:
                    raise ValueError("[QuantLab] QActivation can't initialise clipping bounds from statistics without a strategy!")
                else:
                    if statistics_aware_strategy == 'max':  # look at the minimum and maximum value observed
                        clip_lo, clip_hi = self._qstatistics.min, self._qstatistics.max
                    elif statistics_aware_strategy == 'dist':  # look at the distribution
                        n_std = 3
                        clip_lo = self._qstatistics.mean - n_std * torch.sqrt(self._qstatistics.var)
                        clip_hi = self._qstatistics.mean + n_std * torch.sqrt(self._qstatistics.var)
        else:
            clip_lo = torch.Tensor([-1.0])
            clip_hi = torch.Tensor([1.0])

    @property
    def is_quantising(self) -> bool:
        return self._is_quantising

    def start_quantising(self) -> None:
        self._is_quantising = True

    @classmethod
    def create_from(cls, m: nn.Module):
        raise NotImplementedError


class QATLinear(nn.Module):
    """A base class archetyping QAT-trainable linear operations."""

    def __init__(self,
                 granularity: str):

        super().__init__()

        self._granularity = QGranularity[granularity.upper()]

    @classmethod
    def create_from(cls, m: nn.Module):
        raise NotImplementedError
