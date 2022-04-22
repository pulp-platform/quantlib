import torch
import torch.nn as nn


class EpsTunnel(nn.Module):

    def __init__(self, eps: torch.Tensor):
        """
        The only objects entitled to modify the ``eps_in`` and ``eps_out``
        attributes are ``Rewriter``s.
        """
        super(EpsTunnel, self).__init__()
        self._eps_in  = eps
        self._eps_out = eps

    @property
    def eps_in(self) -> torch.Tensor:
        return self._eps_in

    @property
    def eps_out(self) -> torch.Tensor:
        return self._eps_out

    def set_eps_in(self, eps: torch.Tensor) -> None:
        if not isinstance(eps, torch.Tensor):
            raise TypeError
        elif not eps.shape == self._eps_in.shape:
            raise ValueError
        else:
            self._eps_in = eps

    def set_eps_out(self, eps: torch.Tensor) -> None:
        if not isinstance(eps, torch.Tensor):
            raise TypeError
        elif not eps.shape == self._eps_out.shape:
            raise ValueError
        else:
            self._eps_out = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(self._eps_in != self._eps_out):
            x = self._eps_out * (x / self._eps_in)
        return x
