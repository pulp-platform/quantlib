import torch
import torch.nn as nn
from typing import Tuple, Dict, Union


from quantlib.newquant import resolve_quantspec
from quantlib.newalgorithms.qbase import QATActivation, QATLinear


class TQTActivation(QATActivation):

    def __init__(self,
                 quantspec: Union[Tuple[int, ...], Dict[str, int], str],
                 pinrange: bool,
                 unquant_module: Union[None, nn.Module] = None):

        qrange = resolve_quantspec(quantspec)

        self._n_levels = qrange.n_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the TQT operation to the input array."""
        pass
