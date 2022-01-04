#
# pact_ops.py
#
# Author(s):
# Francesco Conti <f.conti@unibo.it>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
#
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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
from torch import nn
import numpy as np

from .pact_functions import PACTQuantize, TQTQuantize, AlmostSymmQuantFunc, PACTQuantFunc
from .util import assert_param_valid, almost_symm_quant
import math
import copy

from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_registry as sym_registry

from inspect import signature

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
#     'PACTHardSigmoid',
    'PACTIntegerLayerNorm',
    'PACTIntegerGELU',
    'PACTSoftmax',
    'PACTGELU',
    'PACTLayerNorm',
    'PACTIntegerEmbedding',
    'PACTEmbedding',
    'PACTWrapModule',
    'PACTWrapMHSA',
]

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
                    
                    't','t','t',
                    't', 't', 't', 't',
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
                     postattn_requant_mul, postattn_requant_div,
                     wo_requant_mul, wo_requant_div,
                     
                     dim, heads, dim_head,
                     isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2,
                     n_levels):
            
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
                 isoftmaxA, isoftmaxB, isoftmaxC, isoftmaxlog2, n_levels):
        
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
                                 self.n_levels.type_as(q))

class PACTWrapModule(nn.Module):
    
    def __init__(self, module, n_levels, _dict = {}):
        super().__init__()
        self.module = copy.deepcopy(module)
        self.n_levels = n_levels
        self.statTracker = PACTAsymmetricAct(n_levels=n_levels, act_kind='identity', leaky=0., symm=True)
        self._dict = _dict
        
    def forward(self, *x, **kwargs):
        y = self.module.forward(*x, **kwargs)
        self.statTracker(y)
        return y
            
class RequantShift(nn.Module):

    class MyRequantShift(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, mul, add, div, signed, n_levels_out, enable_add_first):
            dummyTwo = torch.tensor(2.).type_as(x)
            dummyOne = torch.tensor(1.).type_as(x)
            dummyZero = torch.tensor(0.).type_as(x)

            y = enable_add_first * (x + torch.round(add/mul)) * mul
            y = y + ((1-enable_add_first) * (x*mul + add))
            
            y = (y/div).round()
            
            x1 = torch.clip(y, dummyZero, n_levels_out-dummyOne).type_as(x)
            
            c = torch.round(n_levels_out/dummyTwo+torch.tensor(0.001).type_as(x)).type_as(x)
            x2 = torch.clip(y, -c, c - ((n_levels_out+dummyOne)%dummyTwo)).type_as(x)
                
            y = (((signed+dummyOne)%dummyTwo)*x1 + signed*x2)
            return y
        
        @staticmethod
        @parse_args('v', 'v', 'v', 't', 't', 't', 't')
        def symbolic(g, x, mul, add, div, signed, n_levels_out, enable_add_first):
#             mul_ = g.op("Constant", value_t=mul)
#             add_ = g.op("Constant", value_t=add)
            div_ = g.op("Constant", value_t=div)
            signed_ = g.op("Constant", value_t=signed)
            n_levels_out_ = g.op("Constant", value_t=n_levels_out)
            
            return g.op("PACTOps::RequantShift", x, mul, add, div_t=div, signed_t=signed, n_levels_t=n_levels_out)
            #return g.op("PACTOps::RequantShift", x)
    
    def __init__(self, mul : torch.Tensor, add : torch.Tensor, n_levels : int, signed : bool = False, D : torch.Tensor = torch.tensor(2**16), enable_add_first=False):
        super(RequantShift, self).__init__()
        self.mul = nn.Parameter(torch.clone(mul).detach())
        self.add = nn.Parameter(torch.clone(add).detach())
        self.div = torch.clone(D.type_as(add)).detach()
        self.signed = torch.Tensor((signed,)).detach()
        self.n_levels_out = torch.Tensor((n_levels,)).detach()
        self.enable_add_first = torch.Tensor((enable_add_first,)).detach()

    def forward(self, x):
        return self.MyRequantShift.apply(x, self.mul.type_as(x), self.add.type_as(x), self.div.type_as(x), self.signed.type_as(x), self.n_levels_out.type_as(x), self.enable_add_first.type_as(x))
    

class PACTEmbedding(torch.nn.Module):

    def __init__(self, n_levels:int = 256, weights : torch.Tensor = torch.Tensor((1.,))):
        super().__init__()
        self.weights = nn.Parameter(weights)
        self.adder = PACTIntegerAdd(n_levels=n_levels, num_args = 2)

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
    
    def __init__(self, n_levels: int = 256, weight : torch.Tensor = torch.Tensor((1.,)), eps_in:float = 1./255, eps_adder:float=1./255, maxval:float=1., twoStage:bool = False):
        super().__init__()
        self.n_levels = n_levels

        self.register_buffer('floor', torch.Tensor((False,)))
        self.register_buffer('clip_gradient', torch.Tensor((True,)))
        self.register_buffer('noisy', torch.Tensor((False,)))
        self.register_buffer('twoStage', torch.Tensor((twoStage,)))

        eps_out = maxval/(self.n_levels//2-1)
        self.eps_out = torch.Tensor((eps_out,))

        # Requantize in two steps - the intermediate step allows for the embedding to have a lower quantization error
        if twoStage:

            clip_lo = -(torch.max(torch.max(torch.abs(weight))))
            clip_hi = AlmostSymmQuantFunc.apply(clip_lo, n_levels)

            eps_weights = (clip_hi-clip_lo)/(n_levels-1)
            eps_bias = eps_weights/eps_adder
            D = 2**16
            
            self.register_buffer('weight', torch.round(PACTQuantize(weight, eps_bias, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / eps_bias))
            self.rqs1 = RequantShift(mul=torch.floor(D*eps_in/eps_adder), add=torch.Tensor((0.,)), signed=True, D=torch.Tensor((D,)), n_levels=n_levels)
            #self.rqs1 = RequantShift(mul=torch.floor(D*eps_in/eps_adder), add=D*self.weight, signed=True, D=torch.Tensor((D,)), n_levels=n_levels)
            self.rqs2 = RequantShift(mul=torch.floor(D*eps_adder/eps_out), add=torch.Tensor((0.,)), signed=True, D=torch.Tensor((D,)), n_levels=n_levels)

        # Requantize in one step - Fewer operations, but the quantization error might be larger
        else:

            clip_lo = -torch.abs(maxval)
            clip_hi = AlmostSymmQuantFunc.apply(clip_lo, n_levels)
            D = 2**16
            
            self.register_buffer('weight', torch.round(PACTQuantize(weight, eps_out/D, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / (eps_out/D)))
            self.rq = RequantShift(mul=torch.floor(D*eps_in/eps_out), add=self.weight, signed=True, D=torch.Tensor((D,)), n_levels=n_levels)
        
    def forward(self, x):
        if self.twoStage:
            out = self.rqs2(self.rqs1(x) + self.weight)
        else:
            out = self.rq(x)
            
        return out

# class PACTHardSigmoid(torch.nn.Module):

#     def __init__(self, n_levels: int = 256):
#         super().__init__()
#         self.n_levels = n_levels
        
#     def forward(self, x):
#         """Approximate Softmax implementation according to the I-BERT paper:
#         https://arxiv.org/abs/2101.01321

#         :param x: 
#         :returns: 
#         :rtype: 

#         """

#         xTilde = (x - torch.max(x))
#         z = torch.floor(-xTilde / math.log(2))
#         p = xTilde + z * math.log(2)
#         y = (0.3585*(p + 1.353)**2 + 0.344) / 2**z
#         ysum = torch.unsqueeze(torch.sum(y, -1), dim=-1)
#         out = y/(ysum)
#         return out

    
class PACTSoftmax(torch.nn.Module):

    def __init__(self, n_levels: int = 256, dim: int = 1):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim
        
    def forward(self, x):
        """Approximate Softmax implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param x: 
        :returns: 
        :rtype: 

        """
        xTilde = (x - torch.max(x, -1, keepdim=True)[0])
        z = torch.floor(-xTilde / math.log(2))
        p = xTilde + z * math.log(2)
        y = (0.3585*(p + 1.353)**2 + 0.344) / 2**z
        ysum = torch.unsqueeze(torch.sum(y, -1), dim=-1)
        out = y/(ysum)
        return out
    
class PACTIntegerSoftmax(torch.nn.Module):

    class MySoftmax(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, log2, coeffA, coeffB, coeffC, n_levels, zero):
            xTilde = (x - torch.max(x, dim=-1, keepdim=True)[0])
            z = torch.floor(-xTilde / log2)
            p = xTilde + z * log2
            y = torch.floor(((coeffA*(p + coeffB)**2 + coeffC)) / 2**(z-8))
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
    
    def __init__(self, n_levels: int = 256, eps_in: float = 1./255):
        super().__init__()

        self.eps_in = eps_in
        self.n_levels = torch.Tensor((n_levels,))
        self.coeffA = torch.Tensor((0.35815147,))
        self.coeffB = torch.Tensor((1.353,))
        self.coeffC = torch.Tensor((0.344,))
        self.log2 = torch.Tensor((1.,))
        self.zero = torch.Tensor((0.,))

        self.updateCoeffs(eps_in)
        
    def updateCoeffs(self, eps):
        """Updates the coefficients, usually only done with the IntegerizeSoftmax pass

        :param eps: Input epsilon
        :returns: 
        :rtype: 

        """

        p = 0
        #eps2 = torch.Tensor((0.35815147 / 2**p,))
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
        
        return self.MySoftmax.apply(x, self.log2.type_as(x), self.coeffA.type_as(x), self.coeffB.type_as(x), self.coeffC.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))
    
class PACTGELU(torch.nn.Module):

    def __init__(self,  n_levels: int = 256):
        super().__init__()

        self.n_levels = n_levels
        
        # Maxval is used to gather statistics
        self.register_buffer('maxval', torch.Tensor((0.,)))
        
    def forward(self, x):
        """Approximate floating point GELU implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param x: 
        :returns: 
        :rtype: 

        """
            
        L = torch.sign(x) * (-0.2888*(torch.clip(torch.abs(x/math.sqrt(2)), min=0, max=1.769) - 1.769)**2 + 1)
        y = x*((1+L)/2)

        self.maxval.data[0] = max(torch.max(torch.abs(y)).item(), self.maxval)

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
    
    def __init__(self, n_levels: int = 256, eps_in = 1.):
        super().__init__()

        self.n_levels = torch.Tensor((n_levels,)).detach()

        self.a = torch.Tensor((-0.288,)).detach()
        self.b = torch.Tensor((-1.769,)).detach()
        self.one = torch.Tensor((1.,)).detach()
        self.sqrttwo = torch.Tensor((4,)).detach()

        self.D = torch.Tensor((2.**24,)).detach()
        self.totScaler = torch.Tensor((255.,)).detach()
        self.maxval = torch.Tensor((0.,)).detach()

        self.zero = torch.Tensor((0.,)).detach()

        self.updateCoeffs(eps_in)
        
    def updateCoeffs(self, eps_in):
        """Updates the polynomial coefficients, usually only done by the IntegerizeGELU pass

        :param eps_in: Input epsilon
        :returns: 
        :rtype: 

        """


        epsX = eps_in * (math.sqrt(2))

        r = 8
        p = 0
        
        #epsB = torch.Tensor((max(epsX, (2*1.769)/(2**r)),))
        epsB = torch.Tensor((epsX,))
        epsA = torch.Tensor(((0.288)/2**p,))
        epsOne = epsB**2*epsA
        epsOut = epsOne * eps_in
        
        self.eps_out = epsOut

        self.a.data[0] = torch.round(-0.288/epsA)
        self.b.data[0] = torch.round(-1.769/epsB)
        self.one.data[0] = torch.round(1./(epsB**2*epsA))
        self.sqrttwo.data[0] = torch.round(torch.Tensor((2.,)))

        self.totScaler.data[0] = torch.round((self.D * self.eps_out) / eps_in)
        
    def forward(self, x):

        """Approximate Integer GELU implementation according to the I-BERT paper:
        https://arxiv.org/abs/2101.01321

        :param eps_in: 
        :returns: 
        :rtype: 

        """
        return self.MyGELU.apply(x, self.b.type_as(x), self.one.type_as(x), self.totScaler.type_as(x), self.D.type_as(x), self.n_levels.type_as(x), self.zero.type_as(x))
        
class PACTIntegerLayerNorm(torch.nn.Module):

    class MyLayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight, bias, D, n_levels):
            nom = x - torch.floor(torch.mean(x, len(x.shape)-1, keepdim=True))
            denom = torch.floor(torch.sqrt(torch.floor(torch.mean(nom**2, len(x.shape)-1, keepdim=True))+1))

            nom = nom * weight

            y = (torch.floor(torch.div(nom,denom)))

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

    
    def __init__(self, n_levels: int = 256, eps_in : float = 1., maxval: float = 1., weight : torch.Tensor = torch.Tensor((1.,)), bias : torch.Tensor = torch.Tensor((0.,)), D=2**24):
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
        
        if not torch.equal(weight, dummyOne) and not torch.equal(bias, dummyZero):
            clip_lo = -max(torch.max(torch.abs(bias)), torch.max(torch.abs(weight)))
            clip_hi = AlmostSymmQuantFunc.apply(clip_lo, n_levels)

            eps_weights = (clip_hi-clip_lo)/(n_levels-1)
            eps_bias = eps_weights
            
            self.eps_weights = eps_weights

            self.weight = nn.Parameter(torch.round(PACTQuantize(weight, eps_weights, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / eps_weights ).detach(), requires_grad=False)
            self.bias = nn.Parameter(torch.round(PACTQuantize(bias, eps_bias, clip_lo, clip_hi, self.floor, self.clip_gradient, self.noisy) / eps_bias).detach(), requires_grad=False)
            self.totScaler = torch.Tensor((torch.round(self.D * (n_levels//2-1)/maxval * eps_weights ),)).detach()

            self.weight *= self.totScaler
            self.bias *= self.totScaler
            
        else:

            self.bias = torch.Tensor((0.,)).detach()
            self.totScaler = torch.Tensor((torch.round(self.D * (n_levels//2-1)/maxval ),)).detach()
            self.weight = self.totScaler
            
    def forward(self, x):
        return self.MyLayerNorm.apply(x, self.weight.type_as(x), self.bias.type_as(x), self.D.type_as(x), self.n_levels.type_as(x))
    
class PACTLayerNorm(torch.nn.Module):

    def __init__(self, n_levels: int = 256, normalized_shape = None, weight = torch.Tensor((1.,)), bias = torch.Tensor((0.,)), D=2**24):
        super().__init__()

        self.n_levels = n_levels

        self.register_buffer('totScaler', torch.Tensor((255.,)))
        self.register_buffer('D', torch.Tensor((D,)))
        self.register_buffer('maxval', torch.Tensor((1.,)))
        self.register_buffer('maxval_tot', torch.Tensor((1.,)))

        self.register_buffer('dummyOne', torch.Tensor((1.,)))
        self.register_buffer('dummyZero', torch.Tensor((0.,)))
        
        self.normalized_shape = normalized_shape

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        
    def forward(self, x):

        nom = x - torch.mean(x, -1, keepdim=True)
        denom = torch.sqrt(torch.mean(torch.pow(nom, 2), -1, keepdim=True)+1e-5)
        y = torch.div(nom,denom)
        
        self.maxval.data[0] = max(torch.max(torch.abs(y)).item(), self.maxval)
        scaler = (self.n_levels)/self.maxval
        self.totScaler.data[0] = math.floor(self.D * scaler)
        
        if not torch.equal(self.weight, self.dummyOne):
            y = y * self.weight
        if not torch.equal(self.weight, self.dummyZero):
            y = y + self.bias
            
        self.maxval_tot.data[0] = max(torch.max(torch.abs(y)).item(), self.maxval)
        
        return y

class PACTUnsignedAct(nn.Module):
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
            n_levels = 256,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            leaky=0.1,
            nb_std=3,
            noisy=False,
            rounding=False,
            tqt=False,
            tqt_beta=0.9,
            tqt_clip_grad=True
    ):

        r"""Constructor.

        :param bits: currently targeted quantization level (default `None`).
        :type  bits: int or float
        :param clip: the value of the clipping factor :math:`\alpha`.
        :type  clip: `torch.Tensor` or float
        :param learn_clip: default `True`; if `False`, do not update the value of the clipping factor `\alpha` with backpropagation.
        :type  learn_clip: bool
        :param act_kind: 'relu', 'relu6', 'leaky_relu'
        :type  act_kind: string
        :param init_clip: 'max' for initialization of clip_hi (on activation of quantization)
                          with max value, 'std' for initialization to mean + nb_std*standard_dev
        :type  init_clip: string
        :param leaky:     leakiness parameter for leaky ReLU activation; unused if act_kind is not 'leaky_relu'
        :param nb_std:    number of standard deviations from mean to initialize the clipping value
        :type  nb_std:    float or int
        """

        super(PACTUnsignedAct, self).__init__()
        act_kind = act_kind.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, act_kind, 'act_kind', ['relu', 'relu6', 'leaky_relu', 'htanh'])
        assert_param_valid(self, init_clip, 'init_clip',  ['max', 'std', 'const'])

        self.tqt = tqt
        self.n_levels = n_levels
        self.clip_hi = torch.nn.Parameter(torch.Tensor((1.,)), requires_grad=learn_clip and not tqt)
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'high':self.clip_hi}
        #if we do TQT, log_t is learned
        if tqt:
            assert (not noisy and learn_clip and rounding), f"PACTUnsignedAct: TQT quantization requires noisy=False, rounding=True, learn_clip=True - you provided noisy={noisy}, rounding={rounding}, learn_clip={learn_clip}"
            #TODO restore this
            self.register_parameter("log_t", nn.Parameter(torch.tensor((0.,)), requires_grad=True))

            self.register_buffer("tqt_beta", torch.tensor(tqt_beta))
            self.register_buffer("tqt_running_beta", torch.tensor(1.))
            self.register_buffer("tqt_running_grad_var", torch.tensor((0.)))
            self.register_buffer("tqt_clip_grad", torch.tensor(tqt_clip_grad))
            self.clipping_params["log_t"] = self.log_t
        else:
            self.tqt_beta = torch.tensor(tqt_beta)
            self.tqt_clip_grad = torch.tensor(tqt_clip_grad)
        self.learn_clip = learn_clip
        self.act_kind = act_kind
        self.init_clip = init_clip
        self.nb_std = nb_std
        self.leaky = leaky
        self.register_buffer('noisy', torch.tensor(noisy))
        self.rounding = rounding
        # this is switched on/off by the PACTActController
        self.register_buffer('started', torch.tensor(False))

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.clip_hi.data),  requires_grad=False)

        self.register_buffer('clip_gradient', torch.tensor(True))
        self.register_buffer('clip_lo', torch.zeros(1))

    def get_eps(self, *args):
        return (self.clip_hi/(self.n_levels-1)).detach().clone()

    def extra_repr(self):
        r = "n_levels={n_levels}, init_clip='{init_clip}', learn_clip={learn_clip}, act_kind='{act_kind}', leaky={leaky}, nb_std={nb_std}".format(**self.__dict__)
        return r

    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.

        See :py:class:`nemo.quant.pact_quant.PACTQuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`

        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """
        # in statistics collection mode, the activation works like a
        # relu/relu6/leaky_relu
        if not self.started:
            x_stat = torch.tensor(x, device=self.max.device, dtype=self.max.dtype) if not isinstance(x, torch.Tensor) else x
            if self.act_kind == 'relu':
                x = torch.nn.functional.relu(x)
            elif self.act_kind == 'relu6':
                x = torch.nn.functional.relu6(x)
            elif self.act_kind == 'leaky_relu':
                x = torch.nn.functional.leaky_relu(x, self.leaky)
            elif self.act_kind == 'htanh':
                x = torch.nn.functional.hardtanh(x)
            with torch.no_grad():
                cur_max = torch.max(x_stat)
                cur_min = torch.min(x_stat)
                self.max.data = torch.maximum(self.max.data, cur_max)
                self.min.data = torch.minimum(self.min.data, cur_min)
                self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * torch.mean(x_stat)
                self.running_var.data = 0.9 * self.running_var.data  + 0.1 * torch.std(x_stat)**2
            return x
        # in normal mode, PACTUnsignedAct uses the PACTQuantFunc
        else:
            eps = self.get_eps()
            if self.tqt:
                #Make sure that the activation is correctly registered with a
                #controller which assigns clip_hi = 2**log_t!
                return TQTQuantize(x, eps, self.log_t, self.clip_lo, self.clip_hi, self.tqt_beta, self.tqt_running_grad_var, self.tqt_running_beta, self.tqt_clip_grad)
            else:
                return PACTQuantize(x, eps, self.clip_lo, self.clip_hi, floor=(not self.rounding), clip_gradient=self.clip_gradient, noisy=self.noisy) # clip_gradient=True keeps NEMO compatibility


class PACTAsymmetricAct(nn.Module):
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
            n_levels=256,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            leaky=0.1,
            symm=False,
            nb_std=3,
            noisy=False,
            rounding=False,
            tqt=False,
            tqt_beta=0.9,
            tqt_clip_grad=True
    ):

        r"""Constructor.
        :param n_levels: number of quantization levels
        :type  n_levels: int
        :param learn_clip: default `True`; if `False`, do not update the value of the clipping factors `\alpha`,`\beta` with backpropagation.
        :type  learn_clip: bool
        :param act_kind: activation type to use in statistics mode
        :type  act_kind: str
        :param symm:     whether or not to enforce (almost-)symmetricity of the clipping range
        :type  symm:     bool
        :param nb_std:   Distance (in number of standard deviations) from mean to set upper/lower clipping bounds if init_clip is 'std'

        """

        super(PACTAsymmetricAct, self).__init__()
        act_kind = act_kind.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, act_kind, 'act_kind', ['identity', 'relu', 'relu6', 'leaky_relu', 'htanh'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'const'])


        self.tqt = tqt
        self.n_levels = n_levels
        self.clip_lo = torch.nn.Parameter(torch.Tensor((-1.,)), requires_grad=learn_clip and not tqt)
        self.clip_hi  = torch.nn.Parameter(torch.Tensor((1.,)),  requires_grad=learn_clip and not symm)
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}
        self.learn_clip = learn_clip
        self.act_kind = act_kind
        self.leaky = leaky
        self.init_clip = init_clip
        self.nb_std = nb_std
        self.symm = symm
        self.register_buffer('noisy', torch.tensor(noisy))
        self.rounding = rounding

        if tqt:
            assert (not noisy and learn_clip and rounding and symm), f"PACTAsymmetricAct: TQT quantization requires noisy=False, rounding=True, learn_clip=True - you provided noisy={noisy}, rounding={rounding}, learn_clip={learn_clip}, symm={symm}"
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
        r = "n_levels={n_levels}, init_clip='{init_clip}', learn_clip={learn_clip}, act_kind='{act_kind}', leaky={leaky}, symm={symm}, nb_std={nb_std}".format(**self.__dict__)
        return r

    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.

        See :py:class:`nemo.quant.pact_quant.PACTQuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`

        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        # in statistics collection mode, the activation works like an identity function (is this intended?)
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
        # in normal mode, PACTUnsignedAct uses
        else:
            eps = self.get_eps()
            if self.tqt:
                #Make sure that the activation is correctly registered with a
                #controller which assigns clip_hi = 2**log_t!
                return TQTQuantize(x, eps, self.log_t, self.clip_lo, self.clip_hi, self.tqt_beta, self.tqt_running_grad_var, self.tqt_running_beta, self.tqt_clip_grad)
            else:
                if self.learn_clip and self.symm:
                    clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
                else:
                    clip_upper = self.clip_hi
                return PACTQuantize(x, eps, self.clip_lo, clip_upper, floor=(not self.rounding), clip_gradient=self.clip_gradient, noisy=self.noisy)

class PACTIntegerConcat(torch.nn.Module):

    def __init__(
            self,
            n_levels=256,
            num_args = 1,
            dim: int = 0,
            stack_flag: bool = False,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            symm=False,
            leaky=0,
            nb_std=3,
            noisy=False,
            rounding=False,
            force_out_eps=False
    ):

        super().__init__()

        self.dim = dim
        self.stack_flag = stack_flag

        self.acts = torch.nn.ModuleList([])
        for i in range(num_args):
            self.acts.append(PACTAsymmetricAct(n_levels=n_levels, init_clip=init_clip, learn_clip=learn_clip, act_kind=act_kind, leaky=leaky, symm=symm, nb_std=nb_std, noisy=noisy, rounding=rounding))

        self.act_out = PACTAsymmetricAct(n_levels=n_levels, init_clip=init_clip, learn_clip=learn_clip, act_kind=act_kind, leaky=leaky, symm=symm, nb_std=nb_std, noisy=noisy, rounding=rounding)

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

    def __init__(
            self,
            n_levels=256,
            num_args = 1,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            symm=False,
            leaky=0,
            nb_std=3,
            noisy=False,
            rounding=False,
            force_out_eps=False
    ):

        super().__init__()
        self.acts = torch.nn.ModuleList([])
        for i in range(num_args):
            self.acts.append(PACTAsymmetricAct(n_levels=n_levels, init_clip=init_clip, learn_clip=learn_clip, act_kind=act_kind, leaky=leaky, symm=symm, nb_std=nb_std, noisy=noisy, rounding=rounding))

        self.act_out = PACTAsymmetricAct(n_levels=n_levels, init_clip=init_clip, learn_clip=learn_clip, act_kind=act_kind, leaky=leaky, symm=symm, nb_std=nb_std, noisy=noisy, rounding=rounding)

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

    def forward(self, *x: torch.Tensor):
        total = self.acts[0](x[0])
        for idx, i in enumerate(x[1:]):
            total = total + self.acts[idx+1](i)
        return total


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


class PACTConv2d(nn.Conv2d):
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
        :param n_levels: Number of weight quantization levels
        :param quantize: how to quantize weights - 'per_layer' or 'per_channel'
        :type  quantize: str
        :param init_clip: how weight clipping parameters should be initialized - 'sawb_symm', 'sawb_asymm', 'max' or 'std'
        :param learn_clip: whether clipping bound(s) should be learned
        :param symm_wts: Indicates that the weights should cover a symmetrical range around 0. If n_levels is an odd number,
               the integer representations of the weights will go from -n_levels/2 to n_levels/2-1, and the clipping range will
               be set accordingly. If init_clip is 'sawb_symm'/'sawb_asymm', the symm_wts parameter has no effect.
        :param kwargs: passed to Conv2d constructor
        # todo: quantize bias??
        """
        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb_symm', 'sawb_asymm', 'const'])
        if init_clip == 'const':
            assert not symm_wts, "PACTConv2d: argument combination init_clip='const' and symm_wts=True not supported!"

        super(PACTConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
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
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip)
        self.register_buffer('clip_gradient', torch.tensor(True))
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        # in the case when learn_clip and symm_wts are both True, clip_hi is not actually used;
        # instead the upper clipping bound is calculated from clip_lo with AlmostSymmQuantFunc.
        # This way, only the lower clip bound is
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=(learn_clip and not symm_wts))
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        if tqt:
            assert (learn_clip and symm_wts), f"PACTConv2d: TQT quantization requires learn_clip=True and symm_wts=True, you provided learn_clip={learn_clip}, symm_wts={symm_wts}"
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

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1, 1))
        return t

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return ((self.clip_hi-self.clip_lo)/(self.n_levels-1)).detach().clone()

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

    def extra_repr(self):
        r = super(PACTConv2d, self).extra_repr()
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

    def forward(self, x):
        if self.started:
            w = self.weight_q
        else:
            w = self.weight

        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

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


class PACTConv1d(nn.Conv1d):
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
        :param n_levels: Number of weight quantization levels
        :param quantize: how to quantize weights - 'per_layer' or 'per_channel'
        :type  quantize: str
        :param init_clip: how weight clipping parameters should be initialized - 'sawb_symm', 'sawb_asymm, 'max' or 'std'
        :param learn_clip: whether clipping bound(s) should be learned
        :param symm_wts: Indicates that the weights should cover a symmetrical range around 0. If n_levels is an odd number,
               the integer representations of the weights will go from -n_levels/2 to n_levels/2-1, and the clipping range will
               be set accordingly. If init_clip is 'sawb_symm'/'sawb_asymm', the symm_wts parameter has no effect.
        :param kwargs: passed to Conv1d constructor
        TODO: implement quantized bias?
        """

        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb_symm', 'sawb_asymm', 'const'])

        super(PACTConv1d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.n_levels = n_levels
        self.quantize = quantize
        self.init_clip = init_clip
        self.learn_clip = learn_clip
        self.symm_wts = symm_wts
        self.nb_std = nb_std
        # this member indicates that quantization is enabled
        self.register_buffer('started', torch.tensor(False))

        clip_lo = torch.tensor(-1.)
        # clip_lo & clip_hi should have dimension (out_channels, 1, 1) to in the case of per-channel quantization.
        # The PACTController will take care of managing them according to the configuration (per-channel, per-layer)
        clip_lo = self.expand_bounds(clip_lo)
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip)
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        # in the case when learn_clip and symm_wts are both True, clip_hi is not actually used;
        # instead the upper clipping bound is calculated from clip_lo with AlmostSymmQuantFunc.
        # This way, only the lower clip bound is
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=(learn_clip and not symm_wts))
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        if tqt:
            assert (learn_clip and symm_wts), f"PACTConv2d: TQT quantization requires learn_clip=True and symm_wts=True, you provided learn_clip={learn_clip}, symm_wts={symm_wts}"
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
        # needed to cleanly call PACTQuantize in all scenarios (CUDA,
        # DataParallel, ...)
        self.register_buffer('clip_gradient', torch.tensor(True))

        self.register_buffer('clip_gradient', torch.tensor(True))

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1))
        return t

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return ((self.clip_hi-self.clip_lo)/(self.n_levels-1)).detach().clone()

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

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
        return (self.weight_q / self.get_eps_w()).round()

    def forward(self, x):
        if self.started:
            w = self.weight_q
        else:
            w = self.weight
        return nn.functional.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


    def extra_repr(self):
        r = super(PACTConv1d, self).extra_repr()
        r +=  f", n_levels={self.n_levels}, quantize='{self.quantize}', init_clip='{self.init_clip}', learn_clip={self.learn_clip}, symm_wts={self.symm_wts}, nb_std={self.nb_std}, tqt={self.tqt}, tqt_beta={self.tqt_beta.item():.2f}, tqt_clip_grad={self.tqt_clip_grad.item()}"
        return r

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


class PACTLinear(nn.Linear):
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
        :param n_levels:      Number of quantization levels
        :param quantize:      quantization type: 'per_layer' or 'per_channel'
        :param init_clip:     how to initialize clipping bounds: 'max', 'std' or 'sawb'
        :param learn_clip:    Whether clipping bound(s) should be learned
        :param symm_wts:      If weights should be forced to be (almost) symmetric around 0 so they map without offset to integers
        :param nb_std:        # of standard deviations from mean to initialize clipping bounds to if init_clip=='std'
        :param kwargs:        passed to nn.Linear constructor
        """

        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb_symm', 'sawb_asymm', 'const'])

        super(PACTLinear, self).__init__(in_features, out_features, **kwargs)
        self.n_levels = n_levels
        self.quantize = quantize
        self.init_clip = init_clip
        self.learn_clip = learn_clip
        self.symm_wts = symm_wts
        self.nb_std = nb_std
        # this member indicates that quantization is enabled
        self.register_buffer('started', torch.tensor(False))

        clip_lo = torch.tensor(-1.)
        clip_lo = self.expand_bounds(clip_lo)
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip)
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=learn_clip and not symm_wts)
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        if tqt:
            assert (learn_clip and symm_wts), f"PACTConv2d: TQT quantization requires learn_clip=True and symm_wts=True, you provided learn_clip={learn_clip}, symm_wts={symm_wts}"
            self.register_parameter("log_t", nn.Parameter(torch.zeros_like(self.clip_lo.data), requires_grad=True))
            self.register_buffer("tqt_beta", torch.tensor(tqt_beta))
            self.register_buffer("tqt_running_beta", torch.tensor(1.))
            self.register_buffer("tqt_running_grad_var",torch.zeros_like(self.clip_lo.data))
            self.register_buffer("tqt_clip_grad", torch.tensor(tqt_clip_grad))
            self.clipping_params["log_t"] = self.log_t
        self.tqt = tqt

        # this member indicates that the module's clipping bounds should not be
        # touched. it is set by the controller
        self.register_buffer('frozen', torch.tensor(False))
        self.register_buffer('clip_gradient', torch.tensor(True))


    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_features * [t])
            t = t.reshape((self.out_features, 1))
        return t

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return ((self.clip_hi-self.clip_lo)/(self.n_levels-1)).detach().clone()

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

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
        return (self.weight_q / self.get_eps_w()).round()

    def forward(self, x):
        if self.started:
            w = self.weight_q
        else:
            w = self.weight
        return nn.functional.linear(x, w, self.bias)


    def extra_repr(self):
        r = super(PACTLinear, self).extra_repr()
        r += ", n_levels={n_levels}, quantize='{quantize}', init_clip='{init_clip}', learn_clip={learn_clip}, symm_wts={symm_wts}, nb_std={nb_std}".format(**self.__dict__)
        return r

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
