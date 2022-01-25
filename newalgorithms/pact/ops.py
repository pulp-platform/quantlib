import torch
import torch.nn as nn
from typing import Tuple, Dict, Union

from quantlib.newutils import quantlib_err_msg
from quantlib.newquant import QRange, resolve_qrangespec
from quantlib.newalgorithms.qbase import QATActivation, QATLinear


class PACTActivation(QATActivation):

    def __init__(self,
                 qrangespec: Union[Tuple[int, ...], Dict[str, int], str],
                 pin_range: bool,
                 unquant_module: nn.Module = nn.Identity()):

        self._qrange = resolve_qrangespec(qrangespec)
        self._pin_range = pin_range  # the specified QRange won't change during training

        if (self.uses_sign_range or self.uses_unsigned) and self._pin_range:
            raise ValueError()

    @property
    def n_levels(self) -> int:
        return self._qrange.n_levels

    @property
    def uses_sign_range(self) -> bool:
        """Is the underlying range the binary sign range {-1, 1}?"""
        return self._qrange.is_sign_range

    @property
    def uses_unsigned(self) -> bool:
        """Is the underlying range a sub-range of some ``UINTB`` range?"""
        return self._qrange.is_unsigned

    @property
    def uses_signed(self) -> bool:
        """Is the underlying range a sub-range of some ``INTB`` range?"""
        return self._qrange.is_quasisymmetric or self._qrange.is_symmetric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the TQT operation to the input array."""
        pass
