#
# pact_ops.py
# Francesco Conti <f.conti@unibo.it>
#
# Copyright (C) 2018-2021 ETH Zurich
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from algorithms.pact.pact_functions import PACT_QuantFunc
import torch

from ..controller import Controller


__all__ = [
    'PACT_UnsignedAct',
    'PACT_AsymmetricAct',
]

class PACT_UnsignedAct(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation, considering unsigned outputs.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`torch.nn.ReLU`, :py:class:`torch.nn.ReLU6` and
    similar activations in a PACT-quantized network.

    This layer can also operate in a special mode, defined by the `statistics` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9

    """

    def __init__(
        self,
        bits=None,
        clip=1.,
        backprop_clip=True,
        act_kind='relu'
    ):

        r"""Constructor.

        :param bits: currently targeted quantization level (default `None`).
        :type  bits: int or float
        :param clip: the value of the clipping factor :math:`\alpha`.
        :type  clip: `torch.Tensor` or float
        :param backprop_clip: default `True`; if `False`, do not update the value of the clipping factor `\alpha` with backpropagation.
        :type  backprop_clip: bool
        :param act_kind: 'relu', 'relu6', 'leaky_relu'
        :type  act_kind: string

        """

        super(PACT_UnsignedAct, self).__init__()
        self.bits = bits
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip = torch.nn.Parameter(torch.Tensor((clip,)).to(device), requires_grad=backprop_clip)
        self.statistics = False
        self.act_kind = act_kind

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.alpha.data).to(device),  requires_grad=False)

    def get_eps(self, *args):
        return self.clip/(2.0**(self.bits)-1)
    
    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.
        
        See :py:class:`nemo.quant.pact_quant.PACT_QuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        # in statistics collection mode, the activation works like a relu/relu6/leaky_relu
        if self.statistics:
            if self.act_kind == 'relu':
                x = torch.nn.functional.relu(x)
            elif self.act_kind == 'relu6':
                x = torch.nn.functional.relu6(x)
            elif self.act_kind == 'leaky_relu':
                x = torch.nn.functional.leaky_relu(x, self.leaky)
            with torch.no_grad():
                self.max[:] = max(self.max.item(), x.max())
                self.min[:] = min(self.min.item(), x.min())
                self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x.mean()
                self.running_var[:]  = 0.9 * self.running_var.item()  + 0.1 * x.std()*x.std()
            return x
        # in normal mode, PACT_UnsignedAct uses 
        else:
            eps = self.get_eps()
            return PACT_QuantFunc(x, eps, 0, self.clip_beta + eps, clip_gradient=True) # clip_gradient=True keeps NEMO compatibility

class PACT_AsymmetricAct(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation, considering signed outputs, not necessarily symmetric.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style quantization functions.

    This layer can also operate in a special mode, defined by the `statistics` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9

    """

    def __init__(
        self,
        bits=None,
        clip_alpha=1.,
        clip_beta=1.,
        backprop_clip=True
    ):

        r"""Constructor.

        :param bits: currently targeted quantization level (default `None`).
        :type  bits: int or float
        :param clip_alpha: the value of the clipping factor :math:`\alpha`.
        :type  clip_alpha: `torch.Tensor` or float
        :param clip_beta: the value of the clipping factor :math:`\beta`.
        :type  clip_beta: `torch.Tensor` or float
        :param backprop_clip: default `True`; if `False`, do not update the value of the clipping factors `\alpha`,`\beta` with backpropagation.
        :type  backprop_clip: bool

        """

        super(PACT_AsymmetricAct, self).__init__()
        self.bits = bits
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_alpha = torch.nn.Parameter(torch.Tensor((clip_alpha,)).to(device), requires_grad=backprop_clip)
        self.clip_beta  = torch.nn.Parameter(torch.Tensor((clip_beta,)).to(device),  requires_grad=backprop_clip)
        self.statistics = False

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.alpha.data).to(device),  requires_grad=False)

    def get_eps(self, *args):
        return (self.clip_alpha+self.clip_beta)/(2.0**(self.bits)-1)
    
    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.
        
        See :py:class:`nemo.quant.pact_quant.PACT_QuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`
        
        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        # in statistics collection mode, the activation works like a relu/relu6/leaky_relu
        if self.statistics:
            with torch.no_grad():
                self.max[:] = max(self.max.item(), x.max())
                self.min[:] = min(self.min.item(), x.min())
                self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x.mean()
                self.running_var[:]  = 0.9 * self.running_var.item()  + 0.1 * x.std()*x.std()
            return x
        # in normal mode, PACT_UnsignedAct uses 
        else:
            eps = self.get_eps()
            return PACT_QuantFunc(x, eps, self.clip_alpha, self.clip_beta + eps)


