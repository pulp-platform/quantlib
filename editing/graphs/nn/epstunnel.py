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

import torch
import torch.nn as nn


class EpsTunnel(nn.Module):

    def __init__(self, eps: torch.Tensor):
        """An object to control the quantisation status of ``torch.Tensor``s.

        When placed downstream with respect to a quantised ``torch.Tensor``,
        this object can be used to:
          * turn a fake-quantised ``torch.Tensor`` into a true-quantised one
            (by setting ``_eps_in`` to the input array's scale and
            ``_eps_out``'s components to one);
          * turn a true-quantised ``torch.Tensor`` into a fake-quantised one
            (by setting ``_eps_in``'s components to one and ``_eps_out`` to
            the desired scale).

        The only objects entitled to modify the ``_eps_in`` and ``_eps_out``
        attributes (i.e., to call the ``set_eps_in`` and ``set_eps_out``
        methods) are ``Rewriter``s.

        """
        super(EpsTunnel, self).__init__()
        self._eps_in  = eps  # TODO: make these attributes "buffers" (so that moving the enclosing `nn.Module` to GPU also affects them)
        self._eps_out = eps

    @property
    def eps_in(self) -> torch.Tensor:
        return self._eps_in

    @property
    def eps_out(self) -> torch.Tensor:
        return self._eps_out

    def set_eps_in(self, eps: torch.Tensor) -> None:

        # validate input
        if not isinstance(eps, torch.Tensor):
            raise TypeError
        elif not eps.shape == self._eps_in.shape:
            raise ValueError

        self._eps_in = eps

    def set_eps_out(self, eps: torch.Tensor) -> None:

        # validate input
        if not isinstance(eps, torch.Tensor):
            raise TypeError
        elif not eps.shape == self._eps_out.shape:
            raise ValueError

        self._eps_out = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(self._eps_in != self._eps_out):
            x = self._eps_out * (x / self._eps_in)
        return x
