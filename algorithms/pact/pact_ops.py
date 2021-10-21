#
# pact_ops.py
#
# Author(s):
# Francesco Conti <f.conti@unibo.it>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
#
# Copyright (c) 2020-2021 ETH Zurich.
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

from typing import Union
import torch
from torch import nn

from .pact_functions import PACTQuantize, TQTQuantize, AlmostSymmQuantFunc, PACTQuantFunc
from .util import assert_param_valid, almost_symm_quant
import math
import copy


__all__ = [
    'PACTUnsignedAct',
    'PACTAsymmetricAct',
    'PACTConv2d',
    'PACTConv1d',
    'PACTLinear',
    'PACTQuantize',
    'PACTIntegerAdd',
    'PACTIntegerConcat',
    'PACTIntegerMatmul',
    'PACTIntegerSoftmax',
    'PACTIntegerLayerNorm',
]

r"""Broadly configurable implementations of the PACT
(https://arxiv.org/pdf/1807.06964) and TQT (https://arxiv.org/abs/1903.08066)
algorithms. These layers require the corresponding `Controller`s to work correctly!

To quantize a given network with these layers, follow these steps:
1. Replace all unsigned activation functions (ReLU, ReLU6, etc) with
`PACTUnsignedAct`, and signed activation functions ([H]TanH, sigmoid etc) with
`PACTAsymmetricAct`, configuring them appropriately (i.e., to replace a HtanH,
you would initialize the `PACTAsymmetricActivation`'s clipping bounds to -1 and
1).
2. Replace all Conv/Linear layers with their quantized counterparts
3. Run the :class:`quantlib.editing.fx.passes.pact.HarmonizePACTNetPass` over
the network if you think it has topological features which require
harmonization (e.g., back-to-back linear layers, branches, etc.)
4. Instantiate :class:`quantlib.algorithms.Controller`s for activations, linear
operators and harmonized operator (i.e., :class:`PACTIntegerAdd` and
:class:`PACTIntegerConcat`) nodes, making sure that all the corresponding nodes
are registered with the correct controller. Note that the
:class:`quantlib.algorithms.pact.PACTActController` must also control the
activations internal to, e.g. :class:`quantlib.algorithms.pact.PACTIntegerAdd`!
The QuantLab project contains an example (MobileNetV2) of how to achieve this
easily and cleanly.
"""

class _PACTActivation(nn.Module):
    # base class to provide common code for PACT Activations
    r"""PACT/TQT activation base class. Before it is 'started', this layer will
        operate in statistics collection mode, in which
        the layer runs in forward-prop without quantization, collecting
        statistics on the activations that can then be used to reset the value
        of :math:`clip_{lo/hi}`. In this mode, the layer collects:
        - tensor-wise maximum value ever seen
        - running average with momentum 0.9
        - running variance with momentum 0.9
        """
    def __init__(self,
                 n_levels : int,
                 init_clip : str,
                 learn_clip : bool,
                 act_kind : str,
                 leaky : float = 0.1,
                 nb_std : Union[float, int] = 3,
                 symm : bool = True,
                 noisy : bool = False,
                 rounding : bool = False,
                 tqt : bool = False,
                 tqt_beta : float = 0.9,
                 tqt_clip_grad : bool = True,
                 signed : bool = True):
        r"""Constructor.

        :param n_levels: currently targeted quantization level (default 256).
        :param init_clip: how the controller should initialize clipping bounds. Can be 'max', 'std' or 'const'.
        :param learn_clip: default `True`; if `False`, do not update the value of the clipping factor(s) with backpropagation.
        :param act_kind: Activation function to use in unquantized mode - can be 'identity', 'relu', 'relu6', 'leaky_relu' or 'htanh'
        :param leaky:     leakiness parameter for leaky ReLU activation; unused if act_kind is not 'leaky_relu'
        :param nb_std:    number of standard deviations from mean to initialize the clipping value
        :param symm:      Whether to enforce symmetric clipping bounds in signed mode, i.e. :math `clip_{hi} = -\frac{n_{levels}-2}{n_{levels}}clip_{lo}`. Unused if `signed` is False
        :param noisy:     whether to add uniform noise with max. amplitude :math `1/2 \epsilon` to the output to simulate additional quantization noise during training. Empirically not very useful.
        :param rounding:  Whether to use rounding (rather than flooring) in the quantization function. This is still integerizable (see paper)!
        :param tqt:       Whether to use the TQT algorithm. Requires `learn_clip=True`, `noisy=False`
        :param tqt_beta:  Momentum for gradient normalization of TQT (see TQT paper).
        :param tqt_clip_grad: Whether to apply the :math `tanh` function to the TQT gradients.
        :param signed:    True if this is a signed activation. The classes `PACTUnsignedActivation` and `PACTAsymmetricActivation
        """
        super(_PACTActivation, self).__init__()
        act_kind = act_kind.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, act_kind, 'act_kind', ['identity', 'relu', 'relu6', 'leaky_relu', 'htanh'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'const'])

        self.tqt = tqt
        self.n_levels = n_levels

        self.clip_hi  = torch.nn.Parameter(torch.Tensor((1.,)),  requires_grad=(learn_clip and not symm) and not tqt)
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'high' : self.clip_hi}
        if signed:
            self.clip_lo = torch.nn.Parameter(torch.Tensor((-1.,)), requires_grad=learn_clip and not tqt)
            self.clipping_params['low'] = self.clip_lo
        else:
            self.register_buffer('clip_lo', torch.zeros(1))

        self.learn_clip = learn_clip
        self.act_kind = act_kind
        self.leaky = leaky
        self.init_clip = init_clip
        self.nb_std = nb_std
        self.symm = symm
        self.register_buffer('noisy', torch.tensor(noisy))
        self.rounding = rounding
        self.signed = signed

        if tqt:
            assert (not noisy and learn_clip and symm), f"{self.__class__.__name__}: TQT quantization requires noisy=False, learn_clip=True - you provided noisy={noisy}, learn_clip={learn_clip}, symm={symm}"
            self.register_parameter("log_t", nn.Parameter(torch.tensor((0.)), requires_grad=True))
            self.register_buffer("tqt_beta", torch.tensor(tqt_beta))
            self.register_buffer("tqt_running_beta", torch.tensor(1.))
            self.register_buffer("tqt_running_grad_var", torch.tensor((0.)))
            self.register_buffer("tqt_clip_grad", torch.tensor(tqt_clip_grad))
            self.clipping_params["log_t"] = self.log_t
        else:
            self.tqt_beta = torch.tensor(tqt_beta)
            self.tqt_clip_grad = torch.tensor(tqt_clip_grad)
        self.tqt = tqt

        # this is switched on/off by the PACTActController
        self.register_buffer('started', torch.tensor(False))

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.clip_hi.data),  requires_grad=False)
        self.register_buffer('clip_gradient', torch.tensor(True))


    def get_eps(self, *args):
        return ((self.clip_hi-self.clip_lo)/(self.n_levels-1)).detach().clone()

    def extra_repr(self):
        r = f"n_levels={self.n_levels}, init_clip='{self.init_clip}', learn_clip={self.learn_clip}, act_kind='{self.act_kind}', leaky={self.leaky}, nb_std={self.nb_std}, tqt={self.tqt}, tqt_beta={self.tqt_beta.item():.2f}, tqt_clip_grad={self.tqt_clip_grad.item()}"
        return r

    def forward(self, x):
        if not self.started:
            x_stat = torch.tensor(x, device=self.max.device, dtype=self.max.dtype) if not isinstance(x, torch.Tensor) else x
            with torch.no_grad():
                self.max[:] = max(self.max.item(), x_stat.max())
                self.min[:] = min(self.min.item(), x_stat.min())
                self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x_stat.mean()
                self.running_var[:]  = 0.9 * self.running_var.item()  + 0.1 * x_stat.std()*x_stat.std()
            if self.act_kind == 'identity':
                return x
            elif self.act_kind == 'relu':
                return torch.nn.functional.relu(x)
            elif self.act_kind == 'relu6':
                return torch.nn.functional.relu6(x)
            elif self.act_kind == 'leaky_relu':
                return torch.nn.functional.leaky_relu(x, self.leaky)
            elif self.act_kind == 'htanh':
                return torch.nn.functional.hardtanh(x)
        else:
            eps = self.get_eps()
            if self.tqt:
                #Make sure that the activation is correctly registered with a
                #controller which assigns clip_hi = 2**log_t!
                return TQTQuantize(x, eps, self.log_t, self.clip_lo, self.clip_hi, self.tqt_beta, self.tqt_running_grad_var, self.tqt_running_beta, self.tqt_clip_grad, self.rounding)
            else:
                if self.learn_clip and self.symm and self.signed:
                    clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
                else:
                    clip_upper = self.clip_hi
                return PACTQuantize(x, eps, self.clip_lo, clip_upper, floor=(not self.rounding), clip_gradient=self.clip_gradient, noisy=self.noisy)


class PACTUnsignedAct(_PACTActivation):
    r"""PACT/TQT activation for unsigned outputs - lower clipping bound is fixed to 0.
    This class is intended to replace ReLU(6), etc. activations in quantized networks.
    Before it's 'started', this layer will collect statistics."""
    def __init__(self, *args, **kwargs):
        super(PACTUnsignedAct, self).__init__(*args, **kwargs, signed=False)


class PACTAsymmetricAct(_PACTActivation):
    r"""PACT/TQT activation, considering signed outputs, not necessarily symmetric.

    Before it's 'started', this layer will collect statistics.
    """

    def __init__(self, *args, **kwargs):
        super(PACTAsymmetricAct, self).__init__(*args, **kwargs, signed=True)

class PACTIntegerConcat(torch.nn.Module):
    r"""Fake-quantized concatenation node. Each input is requantized before being
    concatenated. The
    :class:`quantlib.algorithms.pact.PACTIntegerModulesController` calls the
    :func:`reassign_epsilons` function during every training batch to ensure
    that all input epsilons are identical, forcing each input activation to the
    maximum epsilon of all the inputs. If :func:`torch.stack` is to be used
    instead of :func:`torch.cat`, set `stack_flag` parameter to True.

    """
    def __init__(
            self,
            num_args = 1,
            dim: int = 0,
            stack_flag : bool = False,
            signed : bool = True,
            **kwargs
    ):

        super().__init__()
        self.dim = dim
        self.stack_flag = False

        act_cls = PACTAsymmetricAct if signed else PACTUnsignedAct
        self.acts = torch.nn.ModuleList([])
        for i in range(num_args):
            self.acts.append(act_cls(**kwargs))

        self.clip_lo = self.acts[0].clip_lo
        self.clip_hi = self.acts[0].clip_hi
        self.n_levels = self.acts[0].n_levels

    def reassign_epsilons(self):
        max_clip = -math.inf
        min_clip = math.inf

        for i in self.acts:
            if (i.clip_hi.data - i.clip_lo.data) > (max_clip - min_clip):
                max_clip = i.clip_hi.data
                min_clip = i.clip_lo.data

        # SCHEREMO: This is the part that I might have to think about a bit more...
        for i in self.acts:
            if abs(i.min) < abs(i.max)/2:
                i.symm = False
                i.clip_hi.data = torch.Tensor((max_clip - min_clip,))
                i.clip_lo.data = torch.Tensor((0.,))
            else:
                i.symm = True
                if (abs(min_clip) > max_clip):
                    # Almost symmetrically quantized:
                    i.clip_lo.data, i.clip_hi.data = almost_symm_quant(abs(min_clip), i.n_levels)
                else:
                    # Unsigned quantization
                    i.clip_lo.data, i.clip_hi.data = almost_symm_quant(max_clip/2, i.n_levels)

        self.act_out.eps_in = self.acts[0].get_eps()

    def forward(self, *x):
        if self.stack_flag:
            z = list(map(lambda x: torch.unsqueeze(x, self.dim), x))
        else:
            z = list(x)
        z_act = []
        for idx, i in enumerate(z):
            z_act.append(self.acts[idx](i))
        y = torch.cat(z_act, dim=self.dim)
        return y

class PACTIntegerAdd(torch.nn.Module):
    r"""
    Fake-quantized addition node. Each input is quantized before being added. The
    :class:`quantlib.algorithms.pact.PACTIntegerModulesController` calls the
    :func:`reassign_epsilons` function during every training batch to ensure
    that all input epsilons are identical, forcing each input activation to the
    maximum epsilon of all the inputs.
    """
    def __init__(
            self,
            num_args = 1,
            force_out_eps=False,
            signed : bool = True,
            **kwargs
    ):

        super().__init__()
        act_cls = PACTAsymmetricAct if signed else PACTUnsignedAct
        self.acts = torch.nn.ModuleList([])
        for i in range(num_args):
            self.acts.append(act_cls(**kwargs))

        self.act_out = act_cls(**kwargs)

        self.clip_lo = self.acts[0].clip_lo
        self.clip_hi = self.acts[0].clip_hi
        self.n_levels = self.acts[0].n_levels
        self.force_out_eps = force_out_eps

    def reassign_epsilons(self):
        if not self.force_out_eps:
            max_clip = -math.inf
            min_clip = math.inf
            eps = math.inf

            for i in self.acts:
                if (i.clip_hi.data - i.clip_lo.data) > (max_clip - min_clip):
                    max_clip = i.clip_hi.data
                    min_clip = i.clip_lo.data
                    diff = max_clip - min_clip
                    eps = diff/(self.n_levels-1)

            # SCHEREMO: This is the part that I might have to think about a bit more...
            for i in self.acts:
                # Closer to unsigned than to signed -- Is this reasonable?
                #if abs(i.clip_lo) < abs(i.clip_hi)/2:
                # Make it unsigned if it is only really barely signed... 5 is really arbitrary, though
                if abs(i.clip_lo) < i.get_eps():
                    i.symm = False
                    i.clip_hi.data.copy_(torch.Tensor((eps * (self.n_levels-1),)))
                    i.clip_lo.data.copy_(torch.Tensor((0.,)))
                    # Closer to signed than unsigned
                else:
                    i.symm = True
                    i.clip_lo.data.copy_(torch.Tensor((-(self.n_levels/2)*eps,)))
                    i.clip_hi.data.copy_(torch.Tensor(((self.n_levels/2 - 1)*eps,)))
#                     i.clip_lo.data.copy_(lower_bound)
#                     i.clip_hi.data.copy_(upper_bound)
        else:
            clip_hi = self.act_out.clip_hi.data.detach().clone()
            clip_lo = self.act_out.clip_lo.data.detach().clone()
            for i in self.acts:
                i.clip_hi.data.copy_(clip_hi)
                i.clip_lo.data.copy_(clip_lo)

    def forward(self, *x: torch.Tensor):
        total = self.acts[0](x[0])
        for idx, i in enumerate(x[1:]):
            total = total + self.acts[idx+1](i)
        return self.act_out(total)


class PACTIntegerMatmul(torch.nn.Module):
    def __init__(
            self,
            n_levels=256,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            symm=False,
            leaky=0,
            nb_std=3
    ):

        super().__init__()

    def reassign_epsilons(self):
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mulresult = torch.matmul(x,y)
        return mulresult


class _PACTLinOp:
    # a helper class to provide initialization code common to all PACT/TQT
    # linear operators in the setup_quant_params() function.
    # always call nn.Module.__init__() before calling setup_quant_params()!

    def __init__(self, *args, **kwargs):
        pass


    def setup_quant_params(
            self,
            n_levels : int = 256,
            quantize : str = 'per_layer',
            init_clip : str= 'sawb_asymm',
            learn_clip : bool = False,
            symm_wts : bool = True,
            nb_std : Union[int, float]= 3,
            tqt : bool = False,
            tqt_beta : float = 0.9,
            tqt_clip_grad : bool = True
    ):
        """
        :param n_levels: Number of weight quantization levels
        :param quantize: how to quantize weights - 'per_layer' or 'per_channel'
        :param init_clip: how weight clipping parameters should be initialized - 'sawb_symm', 'sawb_asymm', 'max' or 'std'
        :param learn_clip: whether clipping bound(s) should be learned
        :param symm_wts: Indicates that the weights should cover a symmetrical range around 0. If n_levels is an odd number,
               the integer representations of the weights will go from -n_levels/2 to n_levels/2-1, and the clipping range will
               be set accordingly. If init_clip is 'sawb_symm'/'sawb_asymm', and `learn_clip` or `tqt` are True, the symm_wts parameter has no effect.
        """

        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb_symm', 'sawb_asymm', 'const'])
        if init_clip == 'const':
            assert not symm_wts, f"{self.__class__.__name__}: argument combination init_clip='const' and symm_wts=True not supported!"

        super(_PACTLinOp, self).__init__()
        self.n_levels = n_levels
        self.quantize = quantize
        self.init_clip = init_clip
        self.learn_clip = learn_clip
        # this member indicates that quantization is enabled
        self.register_buffer('started', torch.tensor(False))
        self.symm_wts = symm_wts
        self.nb_std = nb_std
        clip_lo = torch.tensor(-1.)
        # clip_lo & clip_hi should have dimension (out_channels, 1, 1, 1) in case of per-channel quantization.
        # The PACTController will take care of managing them according to the configuration (per-channel, per-layer)
        clip_lo = self.expand_bounds(clip_lo)
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip and not tqt)
        self.register_buffer('clip_gradient', torch.tensor(True))
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        # in the case when learn_clip and symm_wts are both True, clip_hi is not actually used;
        # instead the upper clipping bound is calculated from clip_lo with AlmostSymmQuantFunc.
        # This way, only the lower clip bound is
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=((learn_clip and not tqt) and not symm_wts))
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        if tqt:
            assert (learn_clip and symm_wts), f"{self.__class__.__name__}: TQT quantization requires learn_clip=True and symm_wts=True, you provided learn_clip={learn_clip}, symm_wts={symm_wts}"
            self.register_parameter("log_t", nn.Parameter(torch.zeros_like(self.clip_lo.data), requires_grad=True))
            self.register_buffer("tqt_beta", torch.tensor(tqt_beta))
            self.register_buffer("tqt_running_beta", torch.tensor(1.))
            self.register_buffer("tqt_running_grad_var", torch.zeros_like(self.clip_lo.data))
            self.register_buffer("tqt_clip_grad", torch.tensor(tqt_clip_grad))
            self.clipping_params["log_t"] = self.log_t
        else:
            self.tqt_beta = torch.tensor(tqt_beta)
            self.tqt_clip_grad = torch.tensor(tqt_clip_grad)
        self.tqt = tqt

        # this member indicates that the module's clipping bounds should not be
        # touched. it is set by the controller
        self.register_buffer('frozen', torch.tensor(False))

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return ((self.clip_hi-self.clip_lo)/(self.n_levels-1)).clone().detach()

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

    def extra_repr(self):
        # this might be a little bit dangerous - always inherit from the
        # nn.Module you're extending FIRST and PACTLinOp SECOND
        r = super(self.__class__, self).extra_repr()
        r += f", n_levels={self.n_levels}, quantize='{self.quantize}', init_clip='{self.init_clip}', learn_clip={self.learn_clip}, symm_wts={self.symm_wts}, nb_std={self.nb_std}, tqt={self.tqt}, tqt_beta={self.tqt_beta.item():.2f}, tqt_clip_grad={self.tqt_clip_grad.item()}"
        return r

    @property
    def weight_q(self):
        if not self.tqt:
            if self.learn_clip and self.symm_wts:
                clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
            else:
                clip_upper = self.clip_hi

            return PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, clip_upper, floor=False, clip_gradient=self.clip_gradient)
        else:
            return TQTQuantize(self.weight, self.get_eps_w(), self.log_t, self.clip_lo, self.clip_hi, self.tqt_beta, self.tqt_running_grad_var, self.tqt_running_beta, self.tqt_clip_grad)

    @property
    def weight_int(self):
        return (self.weight_q / self.get_eps_w()).detach().clone().round()


class PACTConv2d(nn.Conv2d, _PACTLinOp):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            n_levels = 256,
            quantize = 'per_layer',
            init_clip = 'sawb_asymm',
            learn_clip = False,
            symm_wts = True,
            nb_std = 3,
            tqt = False,
            tqt_beta = 0.9,
            tqt_clip_grad = True,
            **kwargs
    ):
        """
        :param in_channels: See torch.nn.Conv2d
        :param out_channels: See torch.nn.Conv2d
        :param kernel_size: See torch.nn.Conv2d
        :param kwargs: passed to Conv2d constructor
        """

        super(PACTConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.setup_quant_params(n_levels=n_levels,
                                quantize=quantize,
                                init_clip=init_clip,
                                learn_clip=learn_clip,
                                symm_wts=symm_wts,
                                nb_std=nb_std,
                                tqt=tqt,
                                tqt_beta=tqt_beta,
                                tqt_clip_grad=tqt_clip_grad)
    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1, 1))
        return t


    def forward(self, x):
        if self.started:
            w = self.weight_q
        else:
            w = self.weight

        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


    # this is not very pretty. Any suggestions on how to avoid it are welcome...
    def extra_repr(self):
        return _PACTLinOp.extra_repr(self)

    @classmethod
    def from_conv2d(cls, c : nn.Conv2d, **kwargs):
        # kwargs should be arguments to PACTConv2d
        pact_conv = cls(in_channels=c.in_channels,
                   out_channels=c.out_channels,
                   kernel_size=c.kernel_size,
                   stride=c.stride,
                   padding=c.padding,
                   dilation=c.dilation,
                   groups=c.groups,
                   bias=(c.bias is not None),
                   padding_mode=c.padding_mode,
                   **kwargs)
        # initialize parameters from the nn.Conv2d
        pact_conv.weight.data.copy_(c.weight.data)
        if c.bias is not None:
            pact_conv.bias.data.copy_(c.bias.data)

        return pact_conv


class PACTConv1d(nn.Conv1d, _PACTLinOp):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            n_levels = 256,
            quantize = 'per_layer',
            init_clip = 'sawb_asymm',
            learn_clip = False,
            symm_wts = True,
            nb_std = 3,
            tqt = False,
            tqt_beta = 0.9,
            tqt_clip_grad = True,
            **kwargs
    ):
        """
        :param in_channels: See torch.nn.Conv2d
        :param out_channels: See torch.nn.Conv2d
        :param kernel_size: See torch.nn.Conv2d
        :param kwargs: passed to Conv1d constructor
        """
        super(PACTConv1d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.setup_quant_params(n_levels=n_levels,
                                quantize=quantize,
                                init_clip=init_clip,
                                learn_clip=learn_clip,
                                symm_wts=symm_wts,
                                nb_std=nb_std,
                                tqt=tqt,
                                tqt_beta=tqt_beta,
                                tqt_clip_grad=tqt_clip_grad)

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1))
        return t
    def forward(self, x):
        if self.started:
            w = self.weight_q
        else:
            w = self.weight
        return nn.functional.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return _PACTLinOp.extra_repr(self)

    @classmethod
    def from_conv1d(cls, c : nn.Conv1d, **kwargs):
        # kwargs should be arguments to PACTConv1d
        pact_conv = cls(in_channels=c.in_channels,
                   out_channels=c.out_channels,
                   kernel_size=c.kernel_size,
                   stride=c.stride,
                   padding=c.padding,
                   dilation=c.dilation,
                   groups=c.groups,
                   bias=(c.bias is not None),
                   padding_mode=c.padding_mode,
                   **kwargs)
        # initialize parameters from the nn.Conv1d
        pact_conv.weight.data.copy_(c.weight.data)
        if c.bias is not None:
            pact_conv.bias.data.copy_(c.bias.data)

        return pact_conv




class PACTLinear(nn.Linear, _PACTLinOp):
    def __init__(self,
                 in_features : int,
                 out_features : int,
                 n_levels : int = 256,
                 quantize : str = 'per_layer',
                 init_clip : str = 'sawb_asymm',
                 learn_clip : bool = False,
                 symm_wts : bool = True,
                 nb_std : int = 3,
                 tqt = False,
                 tqt_beta = 0.9,
                 tqt_clip_grad = True,
                 **kwargs):
        """
        :param in_features:   see nn.Linear
        :param out_features:  see nn.Linear
        :param kwargs:        passed to nn.Linear constructor
        """

        super(PACTLinear, self).__init__(in_features, out_features, **kwargs)
        self.setup_quant_params(n_levels=n_levels,
                                quantize=quantize,
                                init_clip=init_clip,
                                learn_clip=learn_clip,
                                symm_wts=symm_wts,
                                nb_std=nb_std,
                                tqt=tqt,
                                tqt_beta=tqt_beta,
                                tqt_clip_grad=tqt_clip_grad)

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_features * [t])
            t = t.reshape((self.out_features, 1))
        return t
    # do not use in training!
    def get_bias_q(self, eps_in):
        # we assume that bias gets quantized to a really high bitwidth so don't
        # clip it
        with torch.no_grad():
            b = PACTQuantize(self.bias, self.get_eps_out(eps_in), -1000.*torch.ones_like(self.clip_lo), 1000.*torch.ones_like(self.clip_hi), clip_gradient=self.clip_gradient)
        return b

    # do not use in training!
    def get_bias_int(self, eps_in):
        return (self.get_bias_q(eps_in)/self.get_eps_out(eps_in)).round()

    def forward(self, x):
        if self.started:
            w = self.weight_q
        else:
            w = self.weight
        return nn.functional.linear(x, w, self.bias)

    def extra_repr(self):
        return _PACTLinOp.extra_repr(self)

    @classmethod
    def from_linear(cls, l : nn.Linear, **kwargs):
        pact_linear = cls(in_features=l.in_features,
                          out_features=l.out_features,
                          bias=(l.bias is not None),
                          **kwargs)
        # initialize parameters from nn.Linear instance
        pact_linear.weight.data.copy_(l.weight.data)
        if l.bias is not None:
            pact_linear.bias.data.copy_(l.bias.data)
        return pact_linear


#############################################################
# THE FOLLOWING CLASSES:
# PACTIntegerLayerNorm, PACTIntegerMatmul, PACTIntegerSoftmax
# ARE STILL IN BETA STADIUM - DO NOT USE FOR IMPORTANT THINGS!
#############################################################

class PACTIntegerLayerNorm(torch.nn.Module):

    def __init__(self, module, n_levels: int = 256):
        super().__init__()

        self.n_levels = n_levels
        self.frozen = False
        self.eps_in = 1.
        self.eps = 1.
        self.module = copy.deepcopy(module)

        self.register_buffer('totScaler', torch.Tensor((255.,)))
        self.register_buffer('D', torch.Tensor((2**16,)))
        self.register_buffer('maxval', torch.Tensor((1.,)))

    def forward(self, x):
        if self.frozen:
            nom = x - torch.floor(torch.mean(x, -1, keepdim=True))
            denom = torch.floor(torch.sqrt(torch.floor(torch.mean(torch.pow(nom, 2), -1, keepdim=True))+self.eps))
            y = torch.floor((torch.floor(torch.div(self.totScaler*nom,denom)))/self.D)
            y = torch.clip(y, -self.n_levels//2, self.n_levels//2-1)
        else:
            y = self.module(x)

            self.maxval.data[0] = max(torch.max(torch.abs(y)).item(), self.maxval)
            scaler = (self.n_levels)/self.maxval
            self.totScaler.data[0] = math.floor(self.D * scaler)

        return y

class PACTIntegerSoftmax(torch.nn.Module):

    def __init__(self, module, eps_in: float = 1./255, n_levels: int = 256):
        super().__init__()
        self.n_levels = n_levels
        self.module = copy.deepcopy(module)
        self.frozen = False
        self.eps_in = eps_in

        self.register_buffer('coeffA', torch.Tensor((0.3585,)))
        self.register_buffer('coeffB', torch.Tensor((1.353,)))
        self.register_buffer('coeffC', torch.Tensor((0.344,)))
        self.register_buffer('log2', torch.Tensor((4.,)))

    def updateCoeffs(self, eps):
        eps2 = (1./(2**8))/(eps**2)

        self.coeffA.data[0] = math.floor(0.3585/eps2)
        self.coeffB.data[0] = math.floor(1.353/eps)
        self.coeffC.data[0] = math.floor(0.344/(eps**2*eps2))
        self.log2.data[0] = 2**math.floor(math.log2(math.log2(2)/(eps)))

    def forward(self, x):
        if self.frozen:
            xTilde = (x - torch.max(x))
            z = torch.floor(-xTilde / self.log2)
            p = xTilde + z * self.log2
            y = (self.coeffA*(p + self.coeffB)**2 + self.coeffC) / 2**z
            ysum = torch.unsqueeze(torch.sum(y, -1), dim=-1)
            out = torch.floor(y*(self.n_levels-1)/ysum)
            return out
        else:
            y = self.module(x)

        return y
