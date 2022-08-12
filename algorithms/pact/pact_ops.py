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

from typing import Union, Optional, Literal
import torch
from torch import nn
import numpy as np

from .pact_functions import PACTQuantize, TQTQuantize, AlmostSymmQuantFunc, PACTQuantFunc
from quantlib.algorithms.generic import CausalConv1d
from .util import assert_param_valid, almost_symm_quant
import math
import copy

from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_registry as sym_registry
from torch.onnx import register_custom_op_symbolic

from inspect import signature

__all__ = [
    '_PACTActivation',
    '_PACTLinOp',
    '_PACTEps',
    'PACTUnsignedAct',
    'PACTAsymmetricAct',
    'PACTConv2d',
    'PACTConv1d',
    'PACTConstWrap',
    'PACTCausalConv1d',
    'PACTLinear',
    'PACTQuantize',
    'TQTQuantize',
    'PACTIntegerAdd',
    'PACTIntegerConcat',
    'PACTIntegerMatmul',
    'PACTIntegerSoftmax',
    'PACTIntegerLayerNorm',
    'PACTIntegerGELU',
    'PACTSoftmax',
    'PACTGELU',
    'PACTLayerNorm',
    'PACTIntegerEmbedding',
    'PACTEmbedding',
    'PACTWrapModule',
    'PACTWrapMHSA',
    'PACTWrapLinearAttention',
    'RequantShift',
    'HardActRequantShift',
    'PACTHardswish',
    'PACTHardsigmoid',
    'PACTIntegerHardswish',
    'PACTIntegerHardsigmoid',
    'PACTMean',
    'PACTDiv',
    'PACTIntegerDiv',
    'PACTExp',
    'PACTIntegerExp',
    'ChannelwiseThreshold'
]

class RequantShift(nn.Module):

    class MyRequantShift(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, mul, add, div, signed, n_levels_out, cmsis_requant):
            # CMSIS-NN performs addition first, then multiplication and
            # division. The rounding in (y / div).round() is folded into the
            # addition.
            if cmsis_requant:
                y = torch.round((x + torch.round((add/mul) + 0.0001)) * mul)
                # Avoid round to even behaviour, friggin pytorch
                y = torch.round((y / div) + 0.0001)
            # PULP-NN performs multiplication first, then addition and
            # division. Division is with flooring.
            else:
                y = x * mul + add
                y = (y/div).floor()

            if not signed:
            # if unsigned: clip y to interval (0, n_levels-1)
                y_tilde =  torch.clip(y, min=torch.zeros(1).type_as(x), max=(n_levels_out-1).type_as(x))
                return y_tilde
            else:
            # if signed: clip y to interval (-n_levels/2, n_levels/2-1)
                c = torch.round(n_levels_out/2. + 0.001)
                # to get correct operators in the exported graph, type_as(x)
                # must be the last thing called on a tensor before feeding into
                # the clip operator. Otherwise, it may get exported as
                # min(max(..)) or some other weirdness
                lo = (c * -1).type_as(y)
                hi = (c-1).type_as(y)

                y_tilde = torch.clip(y, min=lo, max=hi)
                return y_tilde

        @staticmethod
        @parse_args('v', 'v', 'v', 't', 't', 't', 't')
        def symbolic(g, x, mul, add, div, signed, n_levels_out, cmsis_requant):
            signed = torch.Tensor((signed,)).type_as(div)
            div_ = g.op("Constant", value_t=div)
            signed_ = g.op("Constant", value_t=signed)
            n_levels_out_ = g.op("Constant", value_t=n_levels_out)


            output = g.op("PACTOps::RequantShift", x, mul, add, div_t=div, signed_t=signed, n_levels_out_t=n_levels_out)
            #output = g.op("com.microsoft::Gelu", x, mul, add, div_t=div, signed_t=signed, n_levels_out_t=n_levels_out)

            x.setType(x.type())
            output.setType(x.type())
#             output.node().replaceAllUsesWith(g.op("PACTOps::RequantShift", x, mul, add, div_t=div, signed_t=signed, n_levels_out_t=n_levels_out).node())
            return output

    def __init__(self, mul : torch.Tensor, add : torch.Tensor, n_levels : int, signed : bool = False, D : torch.Tensor = torch.Tensor((2**16,)), cmsis_requant=False, requant_node=True):
        super(RequantShift, self).__init__()
        if cmsis_requant:
            add = torch.round(add / mul) * mul
        self.register_buffer('mul', mul.clone().detach())
        self.register_buffer('add', add.clone().detach())
        self.div = D.clone().type_as(add).detach()
        self.signed = signed
        self.n_levels_out = torch.Tensor((n_levels,)).detach()
        # cmsis_requant specifies whether we want to do requantization in
        # CMSIS-NN (true) or PULP-NN (false) style
        self.cmsis_requant = cmsis_requant
        # requant_node specifies whether we want to export a "RequantShift"
        # node in the ONNX graph or explicit mul/add/div operations
        self.requant_node = requant_node

    def forward(self, x):
        if torch.equal(self.mul.type_as(x), self.div.type_as(x)) and torch.equal(self.add.type_as(x), torch.Tensor((0.,)).type_as(x)):
            return x
        if self.requant_node:
            return self.MyRequantShift.apply(x, self.mul.type_as(x), self.add.type_as(x), self.div.type_as(x), self.signed, self.n_levels_out.type_as(x), self.cmsis_requant)
        else:
            # calling `forward` directly does not trigger the symbolic export
            return self.MyRequantShift.forward(None, x, self.mul.type_as(x), self.add.type_as(x), self.div.type_as(x), self.signed, self.n_levels_out, self.cmsis_requant)

class HardActRequantShift(nn.Module):
    #def __init__(self, gamma_h : torch.Tensor, beta_h : torch.Tensor, three :
    #torch.Tensor, six : torch.Tensor, one_over_six : torch.Tensor, D1 : float,
    #D2 : float, hsigmoid : bool, c_lo : torch.Tensor, c_hi : torch.Tensor,
    #eps_half : Optional[float] = None):
    def __init__(self, gamma_h : torch.Tensor, beta_h : torch.Tensor, three : torch.Tensor, six : torch.Tensor, D1 : float, D2 : float, hsigmoid : bool, c_lo : torch.Tensor, c_hi : torch.Tensor, eps_half : Optional[float] = None):
        super(HardActRequantShift, self).__init__()
        self.register_buffer("gamma_h", gamma_h)
        self.register_buffer("beta_h", beta_h)
        self.register_buffer("three", three)
        self.register_buffer("six", six)
        #self.register_buffer("one_over_six", one_over_six)
        self.D1 = D1
        self.D2 = D2
        self.hsigmoid = hsigmoid
        self.shift_factor = D1
        self.c_lo = c_lo
        self.c_hi = c_hi
        if not hsigmoid:
            self.shift_factor = self.shift_factor * D1/D2
        self.register_buffer("eps_half", eps_half)

    def forward(self, x):
        x = x * self.gamma_h.type_as(x)
        x = x + self.beta_h.type_as(x)
        if not self.hsigmoid:
            clip_lo = torch.zeros(1).type_as(x)
            clip_hi = torch.tensor(self.six.item()).type_as(x)
            x1 = x + self.three.type_as(x)
            x1 = torch.clip(x1, clip_lo, clip_hi)
        else:
            x1 = x
        #x1 = x1 * self.one_over_six.type_as(x1)
        if not self.hsigmoid:
            x = x/self.D2
            x = torch.floor(x)
            x1 = x1 * x
            if self.eps_half is not None:
                x1 = x1 + self.eps_half.type_as(x)
        x1 = x1/self.shift_factor.type_as(x)
        x1 = torch.floor(x1)
        clip_lo = torch.tensor(self.c_lo.item()).type_as(x)
        clip_hi = torch.tensor(self.c_hi.item()).type_as(x)
        x1 = torch.clip(x1, clip_lo, clip_hi)
        return x1

class ChannelwiseThreshold(nn.Module):

    class MyChannelwiseThreshold(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, thresh_lo, thresh_hi):
            tmp1 = -1*(x < thresh_lo).type_as(x)
            tmp2 = (x >= thresh_hi).type_as(x)

            return tmp1 + tmp2

        @staticmethod
        @parse_args('v', 't', 't')
        def symbolic(g, x, thresh_lo, thresh_hi):
            thresh_lo_ = g.op("Constant", value_t=thresh_lo)
            thresh_hi_ = g.op("Constant", value_t=thresh_hi)
            return g.op("PACTOps::ChannelwiseThreshold2d", x, thresh_lo_t=thresh_lo, thresh_hi_t=thresh_hi)

    def __init__(self, thresh_lo : torch.Tensor, thresh_hi : torch.Tensor, n_dim : Literal[1,2] = 2):
        super(ChannelwiseThreshold, self).__init__()
        if n_dim == 1:
            thresh_shape = (-1, 1)
        elif n_dim == 2:
            thresh_shape = (-1, 1, 1)
        else:
            assert False, f"ChannelwiseThreshold: n_dim must be 1 or 2, got {n_dim}!"
        self.register_buffer('thresh_lo', thresh_lo.reshape(*thresh_shape).clone().detach())
        self.register_buffer('thresh_hi', thresh_hi.reshape(*thresh_shape).clone().detach())

    def forward(self, x):
        return self.MyChannelwiseThreshold.apply(x, self.thresh_lo.data.type_as(x), self.thresh_hi.data.type_as(x))

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
                 signed : bool = True,
                 upper_percentile : float = 99.5,
                 lower_percentile : float = 0.5,
                 num_bins : int = 2**12 # SCHEREMO: Stay below or at 2**12 - everything else is super slow
                 ):

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
        :param percentile_clipping:    True if the maximum and minimum of a tensor should be determined based on some upper or lower percentile
        :param upper_percentile:    Upper percentile used for percentile-based clipping if percentile_clipping is True
        :param lower_percentile:    Lower percentile used for percentile-based clipping if percentile_clipping is True
        """
        super(_PACTActivation, self).__init__()
        act_kind = act_kind.lower()
        init_clip = init_clip.lower()

        assert_param_valid(self, act_kind, 'act_kind', ['identity', 'relu', 'relu6', 'leaky_relu', 'htanh'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'const', 'klj', 'percentile'])

        self.upper_percentile = upper_percentile/100
        self.lower_percentile = lower_percentile/100

        self.tqt = tqt
        self.n_levels = n_levels

        if act_kind == 'relu6':
            self.clip_hi  = torch.nn.Parameter(torch.Tensor((6.,)),  requires_grad=(learn_clip and not symm) and not tqt)
        else:
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

        self.num_bins = num_bins
        self.register_buffer("histogram",torch.zeros_like(torch.Tensor(range(self.num_bins))))
        self.register_buffer("prevEdges",torch.zeros_like(torch.Tensor(range(self.num_bins+1))))
        self.truemax          = torch.nn.Parameter(torch.Tensor((1,)), requires_grad=False)
        self.truemin          = torch.nn.Parameter(torch.Tensor((-1,)), requires_grad=False)
        self.register_buffer('ready', torch.tensor(False))

    # SCHEREMO: Assume self.histogram magnitude list of data, binned
    def updateHistogram(self, stat):

        def rebinInt(histogram, factor):
            factor = min(self.num_bins//2, factor)
            weight = torch.Tensor([1]*factor).reshape(1,1,-1).type_as(histogram)
            newHistogram = torch.zeros_like(torch.Tensor(range(self.num_bins))).type_as(histogram)
            # Downsample histogram
            res = torch.nn.functional.conv1d(histogram.reshape(1,1,-1), weight, bias=None, stride=factor, padding=0)
            # Set new downsampled histogram in the middle
            newHistogram[(self.num_bins//2 - self.num_bins//(2*factor)):(self.num_bins//2 + self.num_bins//(2*factor))] = res.reshape(-1)
            return newHistogram

        with torch.no_grad():
            # SCHEREMO: get min and max
            newTruemax = max(self.truemax.item(), stat.max())
            newTruemin = min(self.truemin.item(), stat.min())

            # SCHEREMO: only rescale exponentially - allows us to rebin very fast and easy
            binTop = max(abs(self.truemin), self.truemax)
            newBinTop = max(abs(newTruemin), newTruemax)
            expFact = int(max(2**torch.ceil(torch.log2(newBinTop / binTop)),1))
            expTop =  expFact * binTop

            # SCHEREMO: Calculate new histogram according to new range
            addHistogram = torch.histc(input=stat, min=-expTop.item(), max=expTop.item(), bins=self.num_bins)
            resampledHistogram = rebinInt(self.histogram, expFact)

            # SCHEREMO: Add histograms, preserve information about edges
            self.histogram[:] = resampledHistogram + addHistogram
            self.truemax[:] = expTop
            self.truemin[:] = -expTop

    # SCHEREMO: Calculate clipping bounds
    def updateClipBounds(self):
        self.prevEdges[:] = torch.linspace(self.truemin[0], self.truemax[0], self.num_bins+1)
        pdf = self.histogram / (torch.sum(self.histogram))
        weight = torch.Tensor([1]*len(self.histogram.reshape(-1))).reshape(1,1,-1).type_as(self.histogram)
        cdf = torch.nn.functional.conv1d(pdf.reshape(1,1,-1), weight, bias=None, stride=1, padding=(len(self.histogram)-1)).reshape(-1)
        cdf = cdf[:len(self.histogram)]
        rightMinIdx = torch.sum((cdf<self.lower_percentile))
        rightMaxIdx = torch.sum((cdf<self.upper_percentile))
        leftMinIdx = torch.clip(rightMinIdx-1, min=0.)
        leftMaxIdx = torch.clip(rightMaxIdx-1, min=0.)
        #assert rightMaxIdx >= rightMinIdx, "PACTActivation: Clipping bounds swapped!"
        self.min[:] = (self.prevEdges[rightMinIdx]+self.prevEdges[leftMinIdx])/2
        self.max[:] = (self.prevEdges[rightMaxIdx]+self.prevEdges[leftMaxIdx])/2

    def get_eps(self, *args):
        return ((self.clip_hi-self.clip_lo)/(self.n_levels-1)).detach().clone()

    def extra_repr(self):
        r = f"n_levels={self.n_levels}, init_clip='{self.init_clip}', learn_clip={self.learn_clip}, act_kind='{self.act_kind}', leaky={self.leaky}, nb_std={self.nb_std}, tqt={self.tqt}, tqt_beta={self.tqt_beta.item():.2f}, tqt_clip_grad={self.tqt_clip_grad.item()}"
        return r

    def forward(self, x):
        if not self.started:
            if self.act_kind == 'identity':
                res =  x
            elif self.act_kind == 'relu':
                res = torch.nn.functional.relu(x)
            elif self.act_kind == 'relu6':
                res = torch.nn.functional.relu6(x)
            elif self.act_kind == 'leaky_relu':
                res = torch.nn.functional.leaky_relu(x, self.leaky)
            elif self.act_kind == 'htanh':
                res = torch.nn.functional.hardtanh(x)

            x_stat = torch.tensor(res, device=self.max.device, dtype=self.max.dtype) if not isinstance(res, torch.Tensor) else res
            self.updateHistogram(x_stat)

            if self.init_clip == 'percentile' and self.ready:
                res = torch.clip(res, min=self.min, max=self.max)
            else:
                with torch.no_grad():
                    self.min[:] = min(self.min.item(), x_stat.min())
                    self.max[:] = max(self.max.item(), x_stat.max())
                    self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x_stat.mean()
                    self.running_var[:]  = 0.9 * self.running_var.item()  + 0.1 * x_stat.std()*x_stat.std()

            return res

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
        self.stack_flag = stack_flag

        act_cls = PACTAsymmetricAct if signed else PACTUnsignedAct
        self.acts = torch.nn.ModuleList([])
        for i in range(num_args):
            self.acts.append(act_cls(**kwargs))

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
                    #print(diff)
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
            nb_std=3,
            **kwargs
    ):

        super().__init__()

    def reassign_epsilons(self):
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mulresult = torch.matmul(x,y)
        return mulresult

class _PACTEps(nn.Module):
    def __init__(self, startedFlag=False):
        super().__init__()
        self.register_buffer('_eps_in',torch.Tensor((1.,)))
        if startedFlag:
            self.started = False
        self.locked = False

    def set_eps_in(self, eps_in_list):
        self._eps_in[:] = eps_in_list[0]

    def get_eps_in(self):
        return self._eps_in

    @property
    def eps_in(self):
        return self.get_eps_in()

class _PACTLinOp:
    # a helper class to provide initialization code common to all PACT/TQT
    # linear operators in the setup_quant_params() function.
    # always call nn.Module.__init__() before calling setup_quant_params()!

    def __init__(self, *args, **kwargs):
        pass

    def expand_bounds(self, t):
        return t

    # ensure backward compatibility with checkpoints from before the addition
    # of "params_frozen" by adding this as a load_state_dict_pre_hook
    def make_state_dicts_compat(self, state_dict, prefix, _, strict, *args, **kwargs):
        if strict:
            to_fix = ["params", "weight"]
            try:
                if self.bias is not None:
                    to_fix.append("bias")
            except AttributeError:
                pass
            for p in to_fix:
                if prefix+p+"_frozen" not in state_dict.keys():
                    state_dict[prefix+p+"_frozen"] = getattr(self, p+"_frozen")

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
            tqt_clip_grad : bool = True,
            rounding : bool = True
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
        self.rounding = rounding
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
        # this member indicates that parameters (weight + bias) of the layer
        # are frozen
        self.register_buffer('params_frozen', torch.tensor(False))

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

    @property
    def pact_repr_str(self):
        return f", n_levels={self.n_levels}, quantize='{self.quantize}', init_clip='{self.init_clip}', learn_clip={self.learn_clip}, symm_wts={self.symm_wts}, nb_std={self.nb_std}, tqt={self.tqt}, tqt_beta={self.tqt_beta.item():.2f}, tqt_clip_grad={self.tqt_clip_grad.item()}"

    def extra_repr(self):
        # this might be a little bit dangerous - always inherit from the
        # nn.Module you're extending FIRST and PACTLinOp SECOND
        r = super(self.__class__, self).extra_repr()
        r += self.pact_repr_str
        return r

    @property
    def weight_q(self):
        if self.params_frozen:
            wt = self.weight_frozen
        else:
            wt = self.weight
        if not self.tqt:
            if self.learn_clip and self.symm_wts:
                try:
                    clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
                except Exception as e:
                    print(e)
                    import IPython; IPython.embed()
            else:
                clip_upper = self.clip_hi

            return PACTQuantize(wt, self.get_eps_w(), self.clip_lo, clip_upper, floor=False, clip_gradient=self.clip_gradient)
        else:
            return TQTQuantize(wt, self.get_eps_w(), self.log_t, self.clip_lo, self.clip_hi, self.tqt_beta, self.tqt_running_grad_var, self.tqt_running_beta, self.tqt_clip_grad, rounding=True)

    @property
    def weight_int(self):
        return (self.weight_q / self.get_eps_w()).detach().clone().round()

    # not nice: inheriting classes must set up weight_frozen/bias_frozen in
    # their constructors!!!
    def freeze_params(self):
        self.weight_frozen.copy_(self.weight.data)
        if self.bias is not None:
            self.bias_frozen.copy_(self.bias.data)

        self.params_frozen |= True

    def unfreeze_params(self):
        self.params_frozen &= False



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
            rounding = True,
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
        # we want to be able to freeze weights; to avoid updating them, we must
        # keep them in a buffer (e.g., ADAM will keep updating weights even
        # with requires_grad == False)
        self.register_buffer('weight_frozen', self.weight.data.clone())
        if self.bias is not None:
            self.register_buffer('bias_frozen', self.bias.data.clone())
        else:
            self.bias_frozen = None

        self._register_load_state_dict_pre_hook(self.make_state_dicts_compat)


    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1, 1))
        return t


    def forward(self, x):
        b = self.bias
        if self.started:
            w = self.weight_q
        elif self.params_frozen:
            w = self.weight_frozen
            b = self.bias_frozen
        else:
            w = self.weight

        return nn.functional.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


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
            rounding = True,
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
                                tqt_clip_grad=tqt_clip_grad,
                                rounding=rounding)


        self.register_buffer('weight_frozen', self.weight.data.clone())
        if self.bias is not None:
            self.register_buffer('bias_frozen', self.bias.data.clone())
        else:
            self.bias_frozen = None

        self._register_load_state_dict_pre_hook(self.make_state_dicts_compat)

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1))
        return t
    def forward(self, x):
        b = self.bias
        if self.started:
            w = self.weight_q
        elif self.params_frozen:
            w = self.weight_frozen
            b = self.bias_frozen
        else:
            w = self.weight

        return nn.functional.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

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


class PACTCausalConv1d(PACTConv1d, _PACTLinOp):
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
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 1, "Invalid Kernel Size in PACTCausalConv1d: {}".format(kernel_size)
            k = kernel_size[0]
        else:
            k = kernel_size

        if 'dilation' in kwargs.keys():
            dilation = kwargs['dilation']
            if isinstance(dilation, tuple):
                assert len(dilation) == 1, "Invalid Dilation in PACTCausalConv1d: {}".format(dilation)
                dil = dilation[0]
            else:
                dil = dilation
        else:
            dil = 1

        self.__padding = (k-1) * dil

        if 'padding_mode' in kwargs:
            self.padding_mode = kwargs['padding_mode']
        else:
            self.padding_mode = 'zeros'

        super(PACTCausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            n_levels,
            quantize,
            init_clip,
            learn_clip,
            symm_wts,
            nb_std,
            tqt,
            tqt_beta,
            tqt_clip_grad,
            **kwargs)

    def extra_repr(self):
        # done veryy ugly, but I was getting a recursion error all the time and couldn't figure it out
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, n_levels={self.n_levels}, quantize='{self.quantize}', init_clip='{self.init_clip}', learn_clip={self.learn_clip}, symm_wts={self.symm_wts}, nb_std={self.nb_std}, tqt={self.tqt}, tqt_beta={self.tqt_beta.item():.2f}, tqt_clip_grad={self.tqt_clip_grad.item()}"

    def forward(self, input):
        pad_mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
        x = nn.functional.pad(input, (self.__padding, 0), mode=pad_mode)
        result = super(PACTCausalConv1d, self).forward(x)
        return result

    @classmethod
    def from_causalconv1d(cls, c : CausalConv1d, **kwargs):
        # kwargs should be arguments to PACTCausalConv1d
        pact_causalconv = cls(
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            kernel_size=c.kernel_size,
            stride=c.stride,
            dilation=c.dilation,
            groups=c.groups,
            bias=(c.bias is not None),
            padding_mode=c.padding_mode,
            **kwargs)
        # initialize parameters from the nn.Conv1d
        pact_causalconv.weight.data.copy_(c.weight.data)
        if c.bias is not None:
            pact_causalconv.bias.data.copy_(c.bias.data)

        return pact_causalconv


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
                 rounding = True,
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
                                tqt_clip_grad=tqt_clip_grad,
                                rounding=rounding)

        self.register_buffer('weight_frozen', self.weight.data.clone())
        if self.bias is not None:
            self.register_buffer('bias_frozen', self.bias.data.clone())
        else:
            self.bias_frozen = None

        self._register_load_state_dict_pre_hook(self.make_state_dicts_compat)

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
        b = self.bias
        if self.started:
            w = self.weight_q
        elif self.params_frozen:
            w = self.weight_frozen
            b = self.bias_frozen
        else:
            w = self.weight

        return nn.functional.linear(x, w, b)

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
class PACTHardswish(nn.Module):
    def __init__(self, eps_s : float, eps_in_fn : Optional[callable] = None):
        super(PACTHardswish, self).__init__()
        self.eps_s = eps_s
        self.eps_in_fn = eps_in_fn

    def forward(self, x):
        inp = x
        three = torch.tensor(3., dtype=x.dtype, device=x.device)
        six = 2 * three
        z = torch.zeros(1, dtype=x.dtype, device=x.device)
        o = torch.ones(1, dtype=x.dtype, device=x.device)
        # if we have a handle on the input epsilon, quantize the constants 3
        # and 6 to eps_in
        if self.eps_in_fn:
            three = PACTQuantize(three, self.eps_in_fn(), 2., 4., floor=False)
            six = PACTQuantize(six, self.eps_in_fn(), 5., 6., floor=False)
        # now perform quantized hswish with the input data:
        # 1. relu6(x+3)
        x = x + three
        x = torch.minimum(torch.maximum(z, x), six)
        # 2. /6
        one_over_six = PACTQuantize(o/6, self.eps_s, 0., 1., floor=False)
        x = x * one_over_six
        # 3. x * (ans)
        return inp * x

    def get_eps_out(self, eps_in):
        return self.eps_s * eps_in * eps_in


class PACTIntegerHardswish(nn.Module):
    def __init__(self, eps_in : float, eps_s : float):
        super(PACTIntegerHardswish, self).__init__()
        self.eps_in = eps_in
        self.eps_s = eps_s
        three = torch.tensor(3.)
        six = 2 * three
        three_q = torch.round(three/self.eps_in)
        six_q = torch.round(six/self.eps_in)
        self.register_buffer("three", three_q)
        self.register_buffer("six", six_q)
        one_over_six = 1/six
        one_over_six_q = torch.round(one_over_six/eps_s)
        self.register_buffer("one_over_six", one_over_six_q)

    def forward(self, x):
        z = torch.zeros.type_as(x)
        inp = x
        x = x + self.three
        x = torch.clip(x, z, self.six)
        x = x * self.one_over_six
        return inp * x




class PACTHardsigmoid(nn.Module):
    def __init__(self, eps_s : float, eps_in_fn : Optional[callable] = None):
        super(PACTHardsigmoid, self).__init__()
        self.eps_s = eps_s
        self.eps_in_fn = eps_in_fn

    def forward(self, x):
        three = torch.tensor(3., dtype=x.dtype, device=x.device)
        six = 2 * three
        z = torch.zeros(1, dtype=x.dtype, device=x.device)
        o = torch.ones(1, dtype=x.dtype, device=x.device)
        # if we have a handle on the input epsilon, quantize the constants 3
        # and 6 to eps_in
        if self.eps_in_fn:
            three = PACTQuantize(three, self.eps_in_fn(), 2., 4., floor=False)
            six = PACTQuantize(six, self.eps_in_fn(), 5., 6.5, floor=False)
        # now perform quantized hswish with the input data:
        # 1. relu6(x+3)
        x = x + three
        x = torch.minimum(torch.maximum(z, x), six)
        # 2. /6
        one_over_six = PACTQuantize(o/six, self.eps_s, z, o, floor=False)
        return x * one_over_six

    def get_eps_out(self, eps_in):
        return self.eps_s * eps_in


class PACTIntegerHardsigmoid(nn.Module):
    def __init__(self, eps_in : float, eps_s : float):
        super(PACTIntegerHardsigmoid, self).__init__()
        self.eps_in = eps_in
        self.eps_s = eps_s
        three = torch.tensor(3.)
        six = 2 * three
        three_q = torch.round(three/self.eps_in)
        six_q = torch.round(six/self.eps_in)
        self.register_buffer("three", three_q)
        self.register_buffer("six", six_q)
        one_over_six = 1/six
        one_over_six_q = torch.round(one_over_six/eps_s)
        self.register_buffer("one_over_six", one_over_six_q)


    def forward(self, x):
        z = torch.zeros.type_as(x)
        inp = x
        x = x + self.three
        x = torch.clip(x, z, self.six)
        return x * self.one_over_six

class PACTEmbedding(torch.nn.Module):

    def __init__(self, n_levels:int = 256, weights : torch.Tensor = torch.Tensor((1.,)), **kwargs):
        super().__init__()
        self.weights = nn.Parameter(weights)
        self.adder = PACTIntegerAdd(n_levels=n_levels, num_args = 2, **kwargs)

        self.register_buffer('maxval', torch.Tensor((0.,)))

    def reassign_epsilons(self):
        self.adder.reassign_epsilons()

    def forward(self, x):
        out = self.adder(x,self.weights)
        self.maxval.data[0] = max(torch.max(torch.abs(out)).item(), self.maxval)

        return out

class PACTIntegerEmbedding(torch.nn.Module):

    # Implements the integerized version of an Embedding
    # Supports single stage mode, i.e. embeddings are quantized to the output epsilon and double stage mode where weights are quantized to an intermediate epsilon, which is more precise

    def __init__(self, n_levels: int = 256, weight : torch.Tensor = torch.Tensor((1.,)), eps_in:float = 1./255, eps_adder:float=1./255, maxval:float=1., twoStage:bool = False, **kwargs):
        super().__init__()
        self.n_levels = n_levels

        self.register_buffer('floor', torch.Tensor((False,)))
        self.register_buffer('clip_gradient', torch.Tensor((True,)))
        self.register_buffer('noisy', torch.Tensor((False,)))
        self.register_buffer('twoStage', torch.Tensor((twoStage,)))

        eps_out = maxval/(self.n_levels//2-1)
        self.register_buffer("eps_out",torch.Tensor((eps_out,)))

        # Requantize in two steps - the intermediate step allows for the embedding to have a lower quantization error
        if twoStage:

            clip_lo = -(torch.max(torch.max(torch.abs(weight))))
            clip_hi = AlmostSymmQuantFunc.apply(clip_lo, n_levels)

            eps_weights = (clip_hi-clip_lo)/(n_levels-1)
            eps_bias = eps_weights/eps_adder
            D = 2**16

            self.register_buffer('weight', torch.Tensor(torch.round(PACTQuantize(weight, eps_bias, clip_lo,
                                                                    clip_hi, self.floor, self.clip_gradient,
                                                                                 self.noisy) / eps_bias)).detach())
            self.rqs1 = RequantShift(mul=torch.floor(D*eps_in/eps_adder), add=torch.Tensor((0.,)), signed=True, D=torch.Tensor((D,)), n_levels=n_levels, **kwargs)
            #self.rqs1 = Requanl=torch.floor(D*eps_in/eps_adder), add=D*self.weight, signed=True, D=torch.Tensor((D,)), n_levels=n_levels)
            self.rqs2 = RequantShift(mul=torch.floor(D*eps_adder/eps_out), add=torch.Tensor((0.,)), signed=True, D=torch.Tensor((D,)), n_levels=n_levels, **kwargs)

        # Requantize in one step - Fewer operations, but the quantization error might be larger
        else:

            clip_lo = -torch.abs(maxval)
            clip_hi = AlmostSymmQuantFunc.apply(clip_lo, n_levels)
            D = 2**16

            self.register_buffer('weight', torch.round(PACTQuantize(weight, eps_out/D, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / (eps_out/D)).detach())
            self.rq = RequantShift(mul=torch.floor(D*eps_in/eps_out), add=self.weight, signed=True, D=torch.Tensor((D,)), n_levels=n_levels, **kwargs)

    def forward(self, x):
        if self.twoStage:
            out = self.rqs2(self.rqs1(x) + self.weight)
        else:
            out = self.rq(x)

        return out

class PACTExp(torch.nn.Module):

    def __init__(self, n_levels: int = 256, dim: int = 1):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim
        self.coeffA = torch.Tensor((0.35815147,))
        self.coeffB = torch.Tensor((1.353,))
        self.coeffC =  torch.Tensor((0.344,))
        self.log2 =  torch.Tensor((1.,))
        self.clip_gradient = torch.tensor(True)
        self.floor = torch.tensor(False)


    def updateCoeffs(self, eps):
        """Updates the coefficients, usually only done with the IntegerizeSoftmax pass

        :param eps: Input epsilon
        :returns:
        :rtype:

        """

        p = 0
        #eps2 = torch.Tensor((0.35815147 / 2**p,))
        eps = eps
        eps2 = torch.Tensor((0.3585,)).type_as(eps)

        self.coeffA.data[0] = torch.round(0.3585/eps2) * eps2
        self.coeffB.data[0] = torch.round(1.353/eps) * eps
        self.coeffC.data[0] = torch.round(0.344/(eps**2*eps2)) * eps**2*eps2

        #self.log2.data[0] = 2**torch.round(torch.Tensor((math.log2(math.log2(2)/(eps)),)))
        self.log2.data[0] = torch.round(math.log2(2)/(eps)) * eps

    def forward(self, x):
        """Approximate Softmax implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param x:
        :returns:
        :rtype:

        """

#         clip_lo = -torch.abs(torch.max(x))
#         clip_hi = AlmostSymmQuantFunc.apply(clip_lo, self.n_levels)
#         eps = (clip_hi-clip_lo)/self.n_levels

        xTilde = (x - torch.max(x, -1, keepdim=True)[0])
        z = torch.floor(-xTilde / math.log(2))
        p = xTilde + z * math.log(2)
        y = (0.3585*(p + 1.353)**2 + 0.344) / 2**z
        return y

class PACTIntegerExp(torch.nn.Module):

    def __init__(self, n_levels: int = 256, dim: int = 1):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim
        self.register_buffer('coeffA', torch.Tensor((0.35815147,)))
        self.register_buffer('coeffB', torch.Tensor((1.353,)))
        self.register_buffer('coeffC',  torch.Tensor((0.344,)))
        self.register_buffer('log2',  torch.Tensor((1.,)))
        self.register_buffer('clip_gradient', torch.tensor(True))
        self.register_buffer('floor', torch.tensor(False))

    class MyExp(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, log2, coeffA, coeffB, coeffC, n_levels, zero):
            xTilde = (x - torch.max(x, dim=-1, keepdim=True)[0])
            z = torch.floor(-xTilde / log2)
            p = xTilde + z * log2
            y = torch.floor(((coeffA*(p + coeffB)**2 + coeffC)) // (2**z))
            out = torch.clip(y, zero, n_levels-1)

            return out

        @staticmethod
        @parse_args('v', 't', 't', 't', 't', 't', 't')
        def symbolic(g, x, log2, coeffA, coeffB, coeffC, n_levels, zero):
            #return g.op("PACTOps::iSoftmax", x, log2_f = log2, coeffA_f = coeffA, coeffB_f = coeffB, coeffC_f = coeffC, n_levels_f = n_levels)

            log2_ = g.op("Constant", value_t=log2)
            coeffA_ = g.op("Constant", value_t=coeffA)
            coeffB_ = g.op("Constant", value_t=coeffB)
            coeffC_ = g.op("Constant", value_t=coeffC)
            n_levels_ = g.op("Constant", value_t=n_levels)

            return g.op("PACTOps::iExp", x, log2_t=log2, coeffA_t=coeffA, coeffB_t=coeffB,  coeffC_t=coeffC, n_levels_t=n_levels)

    def updateCoeffs(self, eps):
        """Updates the coefficients, usually only done with the IntegerizeSoftmax pass

        :param eps: Input epsilon
        :returns:
        :rtype:

        """

        p = 0
        #eps2 = torch.Tensor((0.35815147 / 2**p,))
        eps = eps
        eps2 = torch.Tensor((0.3585,)).type_as(eps)

        self.coeffA.data[0] = torch.round(0.3585/eps2) * eps2
        self.coeffB.data[0] = torch.round(1.353/eps) * eps
        self.coeffC.data[0] = torch.round(0.344/(eps**2*eps2)) * eps**2*eps2

        #self.log2.data[0] = 2**torch.round(torch.Tensor((math.log2(math.log2(2)/(eps)),)))
        self.log2.data[0] = torch.round(math.log2(2)/(eps)) * eps

    def forward(self, x):
        """Approximate Softmax implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param x:
        :returns:
        :rtype:

        """

        if self.export_node:
            return self.MyExp.apply(x, self.log2.type_as(x), self.coeffA.type_as(x), self.coeffB.type_as(x), self.coeffC.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))
        else:
            return self.MyExp.forward(None, x, self.log2.type_as(x), self.coeffA.type_as(x), self.coeffB.type_as(x), self.coeffC.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))

class PACTSoftmax(_PACTEps):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim
        self.register_buffer('coeffA', torch.Tensor((0.35815147,)))
        self.register_buffer('coeffB', torch.Tensor((1.353,)))
        self.register_buffer('coeffC',  torch.Tensor((0.344,)))
        self.register_buffer('log2',  torch.Tensor((1.,)))
        self.register_buffer('clip_gradient', torch.tensor(True))
        self.register_buffer('floor', torch.tensor(False))

    def set_eps_in(self, eps_list):
        super().set_eps_in(eps_list)
        self.updateCoeffs(self.eps_in)

    def updateCoeffs(self, eps):
        eps2 = torch.Tensor((0.3585,)).type_as(eps)
        self.coeffA.data[0] = torch.round(0.3585/eps2) * eps2
        self.coeffB.data[0] = torch.round(1.353/eps) * eps
        self.coeffC.data[0] = torch.round(0.344/(eps**2*eps2)) * eps**2*eps2
        self.log2.data[0] = torch.round(math.log2(2)/(eps)) * eps

    def forward(self, x):

        def RQ(x, eps):
            if self.started:
                x = torch.floor(x/eps)*eps
            return x

        xTilde = x - RQ(torch.max(x, -1, keepdim=True)[0], self.eps_in)
        z = -RQ(xTilde / self.log2, torch.Tensor((1.,)).type_as(x))
        p = xTilde + z * self.log2
        y = RQ((self.coeffA*(p + self.coeffB)**2 + self.coeffC) / 2**z, self.coeffA*self.eps_in**2)
        ysum = torch.unsqueeze(torch.sum(y, -1), dim=-1)
        out = RQ(y / (ysum), 1./self.n_levels)
        return out

class PACTIntegerSoftmax(torch.nn.Module):

    class MySoftmax(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, log2, coeffA, coeffB, coeffC, n_levels, zero):
            xTilde = (x - torch.max(x, dim=-1, keepdim=True)[0])
            z = torch.floor(-xTilde / log2)
            p = xTilde + z * log2
            y = torch.floor(((coeffA*(p + coeffB)**2 + coeffC)) // (2**z))
            ysum = torch.sum(y, -1, keepdim=True)
            norm = torch.floor(y*(n_levels-1)/(ysum))
            out = torch.clip(norm, zero, n_levels-1)

            return out

        @staticmethod
        @parse_args('v', 't', 't', 't', 't', 't', 't')
        def symbolic(g, x, log2, coeffA, coeffB, coeffC, n_levels, zero):
            #return g.op("PACTOps::iSoftmax", x, log2_f = log2, coeffA_f = coeffA, coeffB_f = coeffB, coeffC_f = coeffC, n_levels_f = n_levels)

            log2_ = g.op("Constant", value_t=log2)
            coeffA_ = g.op("Constant", value_t=coeffA)
            coeffB_ = g.op("Constant", value_t=coeffB)
            coeffC_ = g.op("Constant", value_t=coeffC)
            n_levels_ = g.op("Constant", value_t=n_levels)

            return g.op("PACTOps::iSoftmax", x, log2_t=log2, coeffA_t=coeffA, coeffB_t=coeffB,  coeffC_t=coeffC, n_levels_t=n_levels)

    def __init__(self, n_levels: int = 256, eps_in: float = 1./255, export_node=False):
        super().__init__()

        self.eps_in = eps_in
        self.n_levels = torch.Tensor((n_levels,))
        self.coeffA = torch.Tensor((0.35815147,))
        self.coeffB = torch.Tensor((1.353,))
        self.coeffC = torch.Tensor((0.344,))
        self.log2 = torch.Tensor((1.,))
        self.zero = torch.Tensor((0.,))

        self.updateCoeffs(eps_in)
        self.export_node = export_node

    def updateCoeffs(self, eps):
        """Updates the coefficients, usually only done with the IntegerizeSoftmax pass

        :param eps: Input epsilon
        :returns:
        :rtype:

        """
        eps = eps
        eps2 = torch.Tensor((0.3585,))

        self.coeffA.data[0] = torch.round(0.3585/eps2)
        self.coeffB.data[0] = torch.round(1.353/eps)
        self.coeffC.data[0] = torch.round(0.344/(eps**2*eps2))

        #self.log2.data[0] = 2**torch.round(torch.Tensor((math.log2(math.log2(2)/(eps)),)))
        self.log2.data[0] = torch.round(torch.Tensor((math.log2(2)/(eps)),))

    def forward(self, x):
        """Approximate Softmax implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param x:
        :returns:
        :rtype:

        """
        if self.export_node:
            return self.MySoftmax.apply(x, self.log2.type_as(x), self.coeffA.type_as(x), self.coeffB.type_as(x), self.coeffC.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))
        else:
            return self.MySoftmax.forward(None, x, self.log2.type_as(x), self.coeffA.type_as(x), self.coeffB.type_as(x), self.coeffC.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))

class PACTGELU(_PACTEps):

    def __init__(self):
        super().__init__(True)
        self.register_buffer('a',torch.Tensor((-0.288,)))
        self.register_buffer('b',torch.Tensor((-1.769,)))
        self.register_buffer('one', torch.Tensor((1.,)))
        self.register_buffer('sqrttwo', torch.Tensor((math.sqrt(2),)))

        self.register_buffer('epsA',torch.Tensor((1.,)))
        self.register_buffer('epsB',torch.Tensor((1.,)))
        self.register_buffer('epsOne', torch.Tensor((1.,)))
        self.register_buffer('epsOut', torch.Tensor((1.,)))

    def get_eps_out(self, eps_in):
        self.set_eps_in(eps_in)
        return self.epsOut

    def set_eps_in(self, eps_in_list):
        super().set_eps_in(eps_in_list)
        self.updateCoeffs(self.eps_in)

    def updateCoeffs(self, eps_in):
        def RQ(x,eps):
            if self.started:
                x = torch.round(x/eps)*eps
            return x

        epsX = eps_in / (math.sqrt(2))
        p = 0

        with torch.no_grad():
            epsB = epsX
            epsA = torch.Tensor(((-0.2888)/2**p,)).type_as(eps_in)
            epsOne = epsB**2*epsA
            epsOut = epsOne * eps_in

            self.epsB = epsB
            self.epsA = epsA
            self.epsOne = epsOne
            self.epsOut = epsOut

            self.a = RQ(torch.Tensor((-0.2888,)).type_as(self.a), epsA)
            self.b = RQ(torch.Tensor((-1.769,)).type_as(self.b), epsB)
            self.one = RQ(torch.Tensor((1.,)).type_as(self.one),epsB**2*epsA)
            self.sqrttwo = RQ(torch.Tensor((math.sqrt(2.),)).type_as(self.sqrttwo),eps_in/epsB)

    def forward(self, x):
        def RQ(x,eps):
            if self.started:
                x = torch.floor(x/eps)*eps
            return x

        q = RQ(torch.clip(torch.abs(x/self.sqrttwo),min=torch.Tensor((0,)).type_as(x), max=-self.b), self.epsB)
        L = torch.sign(x) * (self.a * (q + self.b)**2 + self.one)
        y = x * RQ(((self.one+L)/2),self.epsOne)
        return y

class PACTIntegerGELU(torch.nn.Module):

    class MyGELU(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, b, one, totScaler, D, n_levels, zero):
            L = torch.sign(x) * (-(torch.clip(torch.abs(x), min=zero, max=-b) + b)**2 + one)
            y = x*((one+L)>>1)

            y = torch.floor((y * totScaler)/D)
            y = torch.clip(y, min=-n_levels//2, max = n_levels//2-1)

            return y

        @staticmethod
        @parse_args('v', 't','t','t','t','t','t')
        def symbolic(g, x, b, one, totScaler, D, n_levels, zero):

            b_ = g.op("Constant", value_t=b)
            one_ = g.op("Constant", value_t=one)
            totScaler_ = g.op("Constant", value_t=totScaler)
            D_ = g.op("Constant", value_t=D)
            n_levels_ = g.op("Constant", value_t=n_levels)

            return g.op("PACTOps::iGELU", x, b_t=b, one_t=one, totScaler_t=totScaler, D_t=D, n_levels_t=n_levels)

    def __init__(self, n_levels: int = 256, eps_in = 1., maxval = 1., D=2**14, export_node = False):
        super().__init__()

        self.n_levels = torch.Tensor((n_levels,)).detach()

        self.a = torch.Tensor((-0.288,)).detach()
        self.b = torch.Tensor((-1.769,)).detach()
        self.one = torch.Tensor((1.,)).detach()
        self.sqrttwo = torch.Tensor((4,)).detach()

        self.D = torch.Tensor((D,)).detach()
        self.totScaler = torch.Tensor((255.,)).detach()
        self.maxval = torch.Tensor((maxval,)).detach()

        self.zero = torch.Tensor((0.,)).detach()
        self.export_node = export_node

        self.register_buffer("eps_out",torch.Tensor((0,)).detach())

        self.updateCoeffs(eps_in)

    def updateCoeffs(self, eps_in):
        """Updates the polynomial coefficients, usually only done by the IntegerizeGELU pass

        :param eps_in: Input epsilon
        :returns:
        :rtype:

        """
        with torch.no_grad():
            epsX = eps_in * (math.sqrt(2))

            r = 8
            p = 0

            #epsB = torch.Tensor((max(epsX, (2*1.769)/(2**r)),))
            epsB = torch.Tensor((epsX,))
            epsA = torch.Tensor(((0.288)/2**p,))
            epsOne = epsB**2*epsA
            epsOut = epsOne * eps_in

            self.eps_out[0] = self.maxval/(self.n_levels//2-1) #epsOut

            self.a.data[0] = torch.round(-0.288/epsA)
            self.b.data[0] = torch.round(-1.769/epsB)
            self.one.data[0] = torch.round(1./(epsB**2*epsA))
            self.sqrttwo.data[0] = torch.round(torch.Tensor((2.,)))

            self.totScaler.data[0] = torch.round((self.D * epsOut) / self.eps_out)

    def forward(self, x):

        """Approximate Integer GELU implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param eps_in:
        :returns:
        :rtype:

        """
        if self.export_node:
            return self.MyGELU.apply(x, self.b.detach(), self.one.type_as(x), self.totScaler.type_as(x), self.D.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))
        else:
            return self.MyGELU.forward(None, x, self.b.type_as(x), self.one.type_as(x), self.totScaler.type_as(x), self.D.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))

class PACTIntegerLayerNorm(torch.nn.Module):

    class MyLayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight, bias, D, n_levels):
            nom = x - (torch.mean(x, len(x.shape)-1, keepdim=True).int().float())
            denom = torch.floor(torch.sqrt(torch.floor(torch.mean(nom**2, len(x.shape)-1, keepdim=True))+1))

            nom = nom * weight

            y = (torch.div(nom,denom)).int().float()

            y = y + (bias)

            y = torch.floor(y/(D))
            y = torch.clip(y, -n_levels//2, n_levels//2-1)
            return y

        @staticmethod
        @parse_args('v','v','v','t','t')
        def symbolic(g, x, weight, bias, D, n_levels):

            n_levels_ = g.op("Constant", value_t=n_levels)
            D_ = g.op("Constant", value_t=D)

            return g.op("PACTOps::iLayerNorm", x, weight, bias, D_t=D, n_levels_t=n_levels)


    def __init__(self, n_levels: int = 256, eps_in : float = 1., maxval: float = 1., weight : torch.Tensor = torch.Tensor((1.,)), bias : torch.Tensor = torch.Tensor((0.,)), D=2**24, export_node=False, **kwargs):
        super().__init__()

        self.n_levels = torch.Tensor((n_levels,)).detach()

        self.eps = torch.Tensor((eps_in,)).detach()
        self.D = torch.Tensor((D,)).detach()

        # dummyOne and dummyZero are there to have a comparison value on Multi-GPU systems to check if weight and bias are used

        self.floor = torch.Tensor((False,)).detach()
        self.clip_gradient = torch.Tensor((True,)).detach()
        self.noisy = torch.Tensor((False,)).detach()

        # Maxval is used to track statistics
        self.maxval = torch.Tensor((maxval,)).detach()

        dummyOne =  torch.Tensor((1.,)).type_as(weight)
        dummyZero = torch.Tensor((0.,)).type_as(bias)

        self.export_node = export_node

        if not torch.equal(weight, dummyOne) and not torch.equal(bias, dummyZero):
            clip_lo = -max(torch.max(torch.abs(bias)), torch.max(torch.abs(weight)))
            clip_hi = AlmostSymmQuantFunc.apply(clip_lo, n_levels)

            eps_weights = (clip_hi-clip_lo)/(n_levels-1)
            eps_bias = eps_weights

            self.eps_weights =  eps_weights.detach()

            self.register_buffer("weight", torch.Tensor(torch.round(PACTQuantize(weight, eps_weights, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / eps_weights ).detach()))
            self.register_buffer("bias", torch.Tensor(torch.round(PACTQuantize(bias, eps_bias, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / eps_bias).detach()))
            self.register_buffer("totScaler", torch.Tensor((torch.round(self.D * (n_levels//2-1)/maxval * eps_weights ),)).detach())

            self.weight *= self.totScaler
            self.bias *= self.totScaler

        else:

            self.register_buffer("bias", torch.Tensor((0.,)).detach())
            self.register_buffer("totScaler",torch.Tensor((torch.round(self.D * (n_levels//2-1)/maxval ),)).detach())
            self.register_buffer("weight",self.totScaler)

    def forward(self, x):
        if self.export_node:
            return self.MyLayerNorm.apply(x, self.weight.type_as(x), self.bias.type_as(x), self.D.type_as(x), self.n_levels.type_as(x))
        else:
            return self.MyLayerNorm.forward(None, x, self.weight.type_as(x), self.bias.type_as(x), self.D.type_as(x), self.n_levels.type_as(x))

class PACTLayerNorm(_PACTEps, _PACTLinOp):

    def __init__(self, normalized_shape = None, weight = torch.Tensor((1.,)), bias = torch.Tensor((0.,)), eps=1e-5, *args, **kwargs):
        _PACTLinOp.__init__(self)
        _PACTEps.__init__(self)
        self.setup_quant_params(*args, **kwargs)

        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.register_buffer('eps', torch.Tensor((eps,)))
        self.register_buffer('eta', torch.Tensor((1.,)))

        self.div = PACTDiv(Delta=1., stable=False, autoscale=True)

    def get_bias_q(self):
        # we assume that bias gets quantized to a really high bitwidth so don't
        # clip it
        eps = self.div.get_eps_out(self.eps_in*self.get_eps_w(), self.eps_in)
        with torch.no_grad():
            b = PACTQuantize(self.bias, eps, -1000.*torch.ones_like(self.clip_lo), 1000.*torch.ones_like(self.clip_hi), clip_gradient=self.clip_gradient)
        return b

    # do not use in training!
    def get_bias_int(self, eps_in):
        return (self.get_bias_q(eps_in)/self.get_eps_out(eps_in)).round()

    def get_eps_out(self, eps_in):
        self.set_eps_in([eps_in])
        return self.div.get_eps_out(self.eps_in*self.get_eps_w(), self.eps_in)

    def set_eps_in(self, eps_in_list):
        super().set_eps_in(eps_in_list)

        t = self.eps_in / torch.sqrt(self.eps)
        self.eta = torch.ceil(t)

        self.div.set_eps_in([self.eps_in*self.get_eps_w(), self.eps_in])

    def forward(self, x):
        def RQ(x, eps):
            if self.started:
                x = torch.floor(x/eps)*eps
            return x

        nom = x - RQ(torch.mean(x, -1, keepdim=True), self.eps_in)
        var = RQ(torch.mean(torch.pow(nom, 2), -1, keepdim=True), self.eps_in**2 )
        var = var * self.eta**2
        nom = nom * self.eta
        eps = RQ(self.eta**2 * self.eps , self.eps_in**2)

        if self.started:
            assert eps>=self.eps_in**2, "Eps was rounded down in PACTLayerNorm"

        denom = RQ(torch.sqrt(var + eps), self.eps_in)

        if self.started:
            nom = nom*self.weight_q
        else:
            nom = nom*self.weight

        b = self.bias

        y = self.div(nom, denom) + b
        return y

class PACTWrapLinearAttention(nn.Module):
    class LinearAttention(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q,k,v,
                    wq_weight, wq_bias,
                    wk_weight, wk_bias,
                    wv_weight, wv_bias,
                    wo_weight, wo_bias,
                    wq_requant_mul, wq_requant_div,
                    wk_requant_mul, wk_requant_div,
                    wv_requant_mul, wv_requant_div,
                    preattn_requant_mul, preattn_requant_div,
                    normalizer_requant_mul, normalizer_requant_div,
                    postattn_requant_mul, postattn_requant_div,
                    wo_requant_mul, wo_requant_div,
                    dim, heads, dim_head,
                    Delta, eps, act_type,
                    n_levels):
            return q

        @staticmethod
        @parse_args('v','v','v',
                    'v', 'v',
                    'v', 'v',
                    'v', 'v',
                    'v', 'v',

                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',

                    't','t','t',
                    'i', 'i', 'i',
                    't')

        def symbolic(g,
                     q,k,v,
                     wq_weight, wq_bias,
                     wk_weight, wk_bias,
                     wv_weight, wv_bias,
                     wo_weight, wo_bias,
                     wq_requant_mul, wq_requant_div,
                     wk_requant_mul, wk_requant_div,
                     wv_requant_mul, wv_requant_div,
                     preattn_requant_mul, preattn_requant_div,
                     normalizer_requant_mul, normalizer_requant_div,
                     postattn_requant_mul, postattn_requant_div,
                     wo_requant_mul, wo_requant_div,
                     dim, heads, dim_head,
                     Delta, eps, act_type,
                     n_levels):
            return g.op("PACTOps::LinearAttention",
                        q, k, v,
                        wq_weight, wq_bias,
                        wk_weight, wk_bias,
                        wv_weight, wv_bias,
                        wo_weight, wo_bias,
                        wq_requant_mul_t=wq_requant_mul, wq_requant_div_t=wq_requant_div,
                        wk_requant_mul_t=wk_requant_mul, wk_requant_div_t=wk_requant_div,
                        wv_requant_mul_t=wv_requant_mul, wv_requant_div_t=wv_requant_div,
                        wo_requant_mul_t=wo_requant_mul, wo_requant_div_t=wo_requant_div,
                        preattn_requant_mul_t=preattn_requant_mul, preattn_requant_div_t=preattn_requant_div,
                        normalizer_requant_mul_t=normalizer_requant_mul, normalizer_requant_div_t=normalizer_requant_div,
                        postattn_requant_mul_t=postattn_requant_mul, postattn_requant_div_t=postattn_requant_div,
                        dim_t=dim, heads_t=heads, dim_head_t=dim_head,
                        Delta_i = Delta, eps_i = eps, act_type_i=act_type,
                        n_levels_t=n_levels)

    def __init__(self, wq_weight, wq_bias, wq_requant_mul, wq_requant_div,
                 wk_weight, wk_bias, wk_requant_mul, wk_requant_div,
                 wv_weight, wv_bias, wv_requant_mul, wv_requant_div,
                 preattn_requant_mul, preattn_requant_div,
                 normalizer_requant_mul, normalizer_requant_div,
                 postattn_requant_mul, postattn_requant_div,
                 wo_weight, wo_bias, wo_requant_mul, wo_requant_div,
                 dim, heads, dim_head,
                 Delta, eps, act_type,
                 n_levels, linearattention_node=True):
        super().__init__()
        self.wk_weight = nn.Parameter(torch.clone(wk_weight).detach())
        self.wk_bias = nn.Parameter(torch.clone(wk_bias).detach())
        self.wk_requant_mul = torch.clone(wk_requant_mul).detach()
        self.wk_requant_div = torch.clone(wk_requant_div).detach()
        self.wq_weight = nn.Parameter(torch.clone(wq_weight).detach())
        self.wq_bias = nn.Parameter(torch.clone(wq_bias).detach())
        self.wq_requant_mul = torch.clone(wq_requant_mul).detach()
        self.wq_requant_div = torch.clone(wq_requant_div).detach()
        self.wv_weight = nn.Parameter(torch.clone(wv_weight).detach())
        self.wv_bias = nn.Parameter(torch.clone(wv_bias).detach())
        self.wv_requant_mul = torch.clone(wv_requant_mul).detach()
        self.wv_requant_div = torch.clone(wv_requant_div).detach()
        self.wo_weight = nn.Parameter(torch.clone(wo_weight).detach())
        self.wo_bias = nn.Parameter(torch.clone(wo_bias).detach())
        self.wo_requant_mul = torch.clone(wo_requant_mul).detach()
        self.wo_requant_div = torch.clone(wo_requant_div).detach()
        self.preattn_requant_mul = torch.clone(preattn_requant_mul).detach()
        self.preattn_requant_div = torch.clone(preattn_requant_div).detach()
        self.normalizer_requant_mul = torch.clone(normalizer_requant_mul).detach()
        self.normalizer_requant_div = torch.clone(normalizer_requant_div).detach()
        self.postattn_requant_mul = torch.clone(postattn_requant_mul).detach()
        self.postattn_requant_div = torch.clone(postattn_requant_div).detach()
        self.Delta = Delta
        self.eps = eps
        self.act_type = act_type
        self.n_levels = torch.clone(torch.Tensor((n_levels,))).detach()
        self.dim = torch.clone(torch.Tensor((dim,))).detach()
        self.dim_head = torch.clone(torch.Tensor((dim_head,))).detach()
        self.heads = torch.clone(torch.Tensor((heads,))).detach()
        self.linearattention_node = linearattention_node

    def forward(self,q,k,v,**kwargs):
        if self.linearattention_node:
            return self.LinearAttention.apply(q,k,v,
                                              self.wq_weight.type_as(q), self.wq_bias.type_as(q),
                                              self.wk_weight.type_as(q), self.wk_bias.type_as(q),
                                              self.wv_weight.type_as(q), self.wv_bias.type_as(q),
                                              self.wo_weight.type_as(q), self.wo_bias.type_as(q),
                                              self.wq_requant_mul.type_as(q), self.wq_requant_div.type_as(q),
                                              self.wk_requant_mul.type_as(q), self.wk_requant_div.type_as(q),
                                              self.wv_requant_mul.type_as(q), self.wv_requant_div.type_as(q),
                                              self.preattn_requant_mul.type_as(q), self.preattn_requant_div.type_as(q),
                                              self.normalizer_requant_mul.type_as(q), self.normalizer_requant_div.type_as(q),
                                              self.postattn_requant_mul.type_as(q), self.postattn_requant_div.type_as(q),
                                              self.wo_requant_mul.type_as(q), self.wo_requant_div.type_as(q),
                                              self.dim.type_as(q), self.heads.type_as(q), self.dim_head.type_as(q),
                                              self.Delta, self.eps, self.act_type,
                                              self.n_levels.type_as(q))
        else:
            return self.LinearAttention.forward(None, q,k,v,
                                                self.wq_weight.type_as(q), self.wq_bias.type_as(q),
                                                self.wk_weight.type_as(q), self.wk_bias.type_as(q),
                                                self.wv_weight.type_as(q), self.wv_bias.type_as(q),
                                                self.wo_weight.type_as(q), self.wo_bias.type_as(q),
                                                self.wq_requant_mul.type_as(q), self.wq_requant_div.type_as(q),
                                                self.wk_requant_mul.type_as(q), self.wk_requant_div.type_as(q),
                                                self.wv_requant_mul.type_as(q), self.wv_requant_div.type_as(q),
                                                self.preattn_requant_mul.type_as(q), self.preattn_requant_div.type_as(q),
                                                self.normalizer_requant_mul.type_as(q), self.normalizer_requant_div.type_as(q),
                                                self.postattn_requant_mul.type_as(q), self.postattn_requant_div.type_as(q),
                                                self.wo_requant_mul.type_as(q), self.wo_requant_div.type_as(q),
                                                self.dim.type_as(q), self.heads.type_as(q), self.dim_head.type_as(q),
                                                self.Delta, self.eps, self.act_type,
                                                self.n_levels.type_as(q))

class PACTWrapMHSA(nn.Module):

    class MyMHSA(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q,k,v,
                    wq_weight, wq_bias,
                    wk_weight, wk_bias,
                    wv_weight, wv_bias,
                    wo_weight, wo_bias,
                    wq_requant_mul, wq_requant_div,
                    wk_requant_mul, wk_requant_div,
                    wv_requant_mul, wv_requant_div,
                    preattn_requant_mul, preattn_requant_div,
                    postattn_requant_mul, postattn_requant_div,
                    wo_requant_mul, wo_requant_div,
                    dim, heads, dim_head,
                    isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2,
                    n_levels, module):
            return module(q,k,v)


        @staticmethod
        @parse_args('v','v','v',
                    'v', 'v',
                    'v', 'v',
                    'v', 'v',
                    'v', 'v',

                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',
                    't', 't',

                    't','t','t',
                    't', 't', 't', 't',
                    't', 't')
        def symbolic(g,
                     q,k,v,
                     wq_weight, wq_bias,
                     wk_weight, wk_bias,
                     wv_weight, wv_bias,
                     wo_weight, wo_bias,

                     wq_requant_mul, wq_requant_div,
                     wk_requant_mul, wk_requant_div,
                     wv_requant_mul, wv_requant_div,
                     preattn_requant_mul, preattn_requant_div,
                     postattn_requant_mul, postattn_requant_div,
                     wo_requant_mul, wo_requant_div,

                     dim, heads, dim_head,
                     isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2,
                     n_levels, module):

            wk_requant_mul_ = g.op("Constant", value_t=wk_requant_mul)
            wk_requant_div_ = g.op("Constant", value_t=wk_requant_div)
            wq_requant_mul_ = g.op("Constant", value_t=wq_requant_mul)
            wq_requant_div_ = g.op("Constant", value_t=wq_requant_div)
            wv_requant_mul_ = g.op("Constant", value_t=wv_requant_mul)
            wv_requant_div_ = g.op("Constant", value_t=wv_requant_div)
            wo_requant_mul_ = g.op("Constant", value_t=wo_requant_mul)
            wo_requant_div_ = g.op("Constant", value_t=wo_requant_div)
            preattn_requant_mul_ = g.op("Constant", value_t=preattn_requant_mul)
            preattn_requant_div_ = g.op("Constant", value_t=preattn_requant_div)
            postattn_requant_mul_ = g.op("Constant", value_t=postattn_requant_mul)
            postattn_requant_div_ = g.op("Constant", value_t=postattn_requant_div)
            isoftmaxA_ = g.op("Constant", value_t=isoftmaxA)
            isoftmaxB_ = g.op("Constant", value_t=isoftmaxB)
            isoftmaxC_ = g.op("Constant", value_t=isoftmaxC)
            isoftmaxlog2_ = g.op("Constant", value_t=isoftmaxlog2)
            n_levels_ = g.op("Constant", value_t=n_levels)
            dim_ = g.op("Constant", value_t=dim)
            dim_head_ = g.op("Constant", value_t=dim_head)
            heads_ = g.op("Constant", value_t=heads)

            return g.op("PACTOps::MultiHeadSelfAttention",
                        q, k, v,
                        wq_weight, wq_bias,
                        wk_weight, wk_bias,
                        wv_weight, wv_bias,
                        wo_weight, wo_bias,
                        wq_requant_mul_t=wq_requant_mul, wq_requant_div_t=wq_requant_div,
                        wk_requant_mul_t=wk_requant_mul, wk_requant_div_t=wk_requant_div,
                        wv_requant_mul_t=wv_requant_mul, wv_requant_div_t=wv_requant_div,
                        wo_requant_mul_t=wo_requant_mul, wo_requant_div_t=wo_requant_div,
                        preattn_requant_mul_t=preattn_requant_mul, preattn_requant_div_t=preattn_requant_div,
                        postattn_requant_mul_t=postattn_requant_mul, postattn_requant_div_t=postattn_requant_div,
                        dim_t=dim, heads_t=heads, dim_head_t=dim_head,
                        isoftmaxA_t=isoftmaxA, isoftmaxB_t=isoftmaxB, isoftmaxC_t=isoftmaxC, isoftmaxlog2_t=isoftmaxlog2,
                        n_levels_t=n_levels)


    def __init__(self,
                 wq_weight, wq_bias, wq_requant_mul, wq_requant_div,
                 wk_weight, wk_bias, wk_requant_mul, wk_requant_div,
                 wv_weight, wv_bias, wv_requant_mul, wv_requant_div,
                 preattn_requant_mul, preattn_requant_div,
                 postattn_requant_mul, postattn_requant_div,
                 wo_weight, wo_bias, wo_requant_mul, wo_requant_div,
                 dim, heads, dim_head,
                 isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2, n_levels, module):

        super().__init__()
        self.module = copy.deepcopy(module)
        self.wk_weight = nn.Parameter(torch.clone(wk_weight).detach())
        self.wk_bias = nn.Parameter(torch.clone(wk_bias).detach())
        self.wk_requant_mul = torch.clone(wk_requant_mul).detach()
        self.wk_requant_div = torch.clone(wk_requant_div).detach()
        self.wq_weight = nn.Parameter(torch.clone(wq_weight).detach())
        self.wq_bias = nn.Parameter(torch.clone(wq_bias).detach())
        self.wq_requant_mul = torch.clone(wq_requant_mul).detach()
        self.wq_requant_div = torch.clone(wq_requant_div).detach()
        self.wv_weight = nn.Parameter(torch.clone(wv_weight).detach())
        self.wv_bias = nn.Parameter(torch.clone(wv_bias).detach())
        self.wv_requant_mul = torch.clone(wv_requant_mul).detach()
        self.wv_requant_div = torch.clone(wv_requant_div).detach()
        self.wo_weight = nn.Parameter(torch.clone(wo_weight).detach())
        self.wo_bias = nn.Parameter(torch.clone(wo_bias).detach())
        self.wo_requant_mul = torch.clone(wo_requant_mul).detach()
        self.wo_requant_div = torch.clone(wo_requant_div).detach()
        self.preattn_requant_mul = torch.clone(preattn_requant_mul).detach()
        self.preattn_requant_div = torch.clone(preattn_requant_div).detach()
        self.postattn_requant_mul = torch.clone(postattn_requant_mul).detach()
        self.postattn_requant_div = torch.clone(postattn_requant_div).detach()
        self.isoftmaxA = torch.clone(isoftmaxA).detach()
        self.isoftmaxB = torch.clone(isoftmaxB).detach()
        self.isoftmaxC = torch.clone(isoftmaxC).detach()
        self.isoftmaxlog2 = torch.clone(isoftmaxlog2).detach()
        self.n_levels = torch.clone(torch.Tensor((n_levels,))).detach()
        self.dim = torch.clone(torch.Tensor((dim,))).detach()
        self.dim_head = torch.clone(torch.Tensor((dim_head,))).detach()
        self.heads = torch.clone(torch.Tensor((heads,))).detach()

    def forward(self, q,k,v, **kwargs):
        return self.MyMHSA.apply(q,k,v,
                                 self.wq_weight.type_as(q), self.wq_bias.type_as(q),
                                 self.wk_weight.type_as(q), self.wk_bias.type_as(q),
                                 self.wv_weight.type_as(q), self.wv_bias.type_as(q),
                                 self.wo_weight.type_as(q), self.wo_bias.type_as(q),
                                 self.wq_requant_mul.type_as(q), self.wq_requant_div.type_as(q),
                                 self.wk_requant_mul.type_as(q), self.wk_requant_div.type_as(q),
                                 self.wv_requant_mul.type_as(q), self.wv_requant_div.type_as(q),
                                 self.preattn_requant_mul.type_as(q), self.preattn_requant_div.type_as(q),
                                 self.postattn_requant_mul.type_as(q), self.postattn_requant_div.type_as(q),
                                 self.wo_requant_mul.type_as(q), self.wo_requant_div.type_as(q),
                                 self.dim.type_as(q), self.heads.type_as(q), self.dim_head.type_as(q),
                                 self.isoftmaxA.type_as(q), self.isoftmaxB.type_as(q), self.isoftmaxC.type_as(q), self.isoftmaxlog2.type_as(q),
                                 self.n_levels.type_as(q), self.module)

class PACTDiv(_PACTEps):
    def __init__(self, Delta=2**8, stable=True, eps_div = 1e-5, autoscale=True):

        assert not ((stable) and (eps_div <= 0.) ), "Either stabilize division and choose eps > 0 or don't stabilize!"

        super().__init__(True)

        self.Delta = Delta
        self.stable = stable
        self.autoscale = autoscale
        self.register_buffer('running_min' , torch.Tensor((1.,)))

        self.register_buffer('eta', torch.Tensor((1.,)))
        self.register_buffer('eps_div', torch.Tensor((eps_div,)))
        self.register_buffer('eps_in_x', torch.Tensor((eps_div,)))
        self.register_buffer('eps_in_y', torch.Tensor((eps_div,)))

        self.set_eps_in([torch.Tensor((eps_div,)), torch.Tensor((eps_div,))])

    def set_eps_in(self, eps_in_list):
        self.eps_in_x[:] = eps_in_list[0]
        self.eps_in_y[:] = eps_in_list[1]
        assert self.eps_in_y > 0, "PACTDiv: Denominator's eps is lt or eq 0!"

        if self.stable:
            t = self.eps_in_y / self.eps_div
            self.eta = torch.ceil(t)

    def get_eps_out(self, eps_in_x, eps_in_y):
        self.set_eps_in([eps_in_x, eps_in_y])
        if not self.locked:
            return eps_in_x / eps_in_y
        else:
            return eps_in_x / (eps_in_y*self.Delta)

    def forward(self,x,y):

        def RQ(x, eps):
            if self.started:
                x = torch.floor(x / eps) * eps
            return x

        if self.stable:
            y = y * self.eta
            x = x * self.eta
            eps = RQ(self.eta * self.eps_div, self.eps_in_y)
            if self.started:
                assert eps>=self.eps_in_y, "Eps was rounded down in PACTDiv"

        else:
            eps = 0

        y = y + eps
        assert torch.sum( y > 0 ) == len(y.reshape(-1)), "PACTDiv: Dividing by negative numbers not allowed!"
        with torch.no_grad():
            if self.autoscale:
                x_hat = x/self.eps_in_x
                y_hat = y/self.eps_in_y
                div = torch.abs((x_hat)/(y_hat))
                self.running_min = torch.min(self.running_min, torch.min(torch.where(div > 0, div, div.max())))
                self.Delta = torch.abs(torch.ceil(self.running_min**(-1)))

        if not self.locked:
            return RQ(x/y , self.get_eps_out(self.eps_in_x, self.eps_in_y))
        else:
            return RQ(x*self.Delta/y , self.get_eps_out(self.eps_in_x, self.eps_in_y))

class PACTIntegerDiv(nn.Module):

    class MyIntegerDiv(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, y, Delta, eps):
            y = y + eps
            return torch.floor(x * Delta / (y))

        @staticmethod
        @parse_args('v','v','i', 'i')
        def symbolic(g, x, y, Delta, eps):
            return g.op("PACTOps::IntegerDiv", x, y, Delta_i = Delta, eps_i=eps)

    def __init__(self, Delta, integer_node=True, eps=1):
        super().__init__()
        self.Delta = Delta
        self.eps = eps
        self.integer_node = integer_node

    def forward(self,x,y):
        if self.integer_node:
            return self.MyIntegerDiv.apply(x,y,self.Delta,self.eps)
        else:
            return self.MyIntegerDiv.forward(None, x,y,self.Delta,self.eps)

class PACTConstWrap(nn.Module):
    def __init__(self, eps=1.):
        super().__init__()
        self.register_buffer('eps', torch.Tensor((eps,)))

    def set_eps(self, eps):
        self.eps = torch.Tensor((eps,)).type_as(self.eps)

    def forward(self, x):
        with torch.no_grad():
            self.set_eps(x)
        return self.eps

class PACTMean(_PACTEps):
    def __init__(self):
        super().__init__(True)

    def forward(self, x, **kwargs):
        def RQ(x, eps):
            if self.started:
                x = torch.floor(x/eps)*eps
            return x

        return RQ(torch.mean(x, **kwargs), self.eps_in)

class PACTWrapModule(nn.Module):

    def __init__(self, module, n_levels, _dict = {}, quantize : bool = False, **actArgs):
        super().__init__()

        default_kwargs = {'learn_clip': True, 'init_clip': 'max', 'act_kind': 'identity', 'leaky': 0.0}
        default_kwargs.update(actArgs)

        self.n_levels = n_levels
        self._dict = _dict
        self.quantize = quantize
        self.actArgs = actArgs

        self.module = copy.copy(module)
        self.statTracker = PACTAsymmetricAct(n_levels=n_levels, **default_kwargs)

    def forward(self, *x, **kwargs):
        y = self.module.forward(*x, **kwargs)
        z = self.statTracker(y)
        if self.quantize:
            return z
        else:
            return y
