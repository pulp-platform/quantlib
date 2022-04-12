#
# __init__.py
#
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

import torch

from quantlib.algorithms.ana.lib import ana_normal, ana_logistic, ana_uniform, ana_triangular


try:
    import ana_uniform_cuda
    import ana_triangular_cuda
    import ana_normal_cuda
    import ana_logistic_cuda
    use_ana_cuda_kernels = True
except ImportError:
    use_ana_cuda_kernels = False


__all__ = [
    'ANAUniform',
    'ANATriangular',
    'ANANormal',
    'ANALogistic',
]


# uniform noise

class ANAUniform(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and uniform noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, mi, sigma, strategy, training):
        ctx.save_for_backward(x_in, q, t, mi, sigma)
        if use_ana_cuda_kernels and x_in.is_cuda:
            x_out = ana_uniform_cuda.forward(x_in, q, t, mi, sigma, strategy, torch.Tensor([training]).to(sigma))
        else:
            x_out = ana_uniform.forward(x_in, q, t, mi, sigma, strategy, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, mi, sigma = ctx.saved_tensors
        if use_ana_cuda_kernels and grad_in.is_cuda:
            grad_out = ana_uniform_cuda.backward(grad_in, x_in, q, t, mi, sigma)
        else:
            grad_out = ana_uniform.backward(grad_in, x_in, q, t, mi, sigma)
        return grad_out, None, None, None, None, None, None


# triangular noise

class ANATriangular(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and triangular noise
    on the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, mi, sigma, strategy, training):
        ctx.save_for_backward(x_in, q, t, mi, sigma)
        if use_ana_cuda_kernels and x_in.is_cuda:
            x_out = ana_triangular_cuda.forward(x_in, q, t, mi, sigma, strategy, torch.Tensor([training]).to(sigma))
        else:
            x_out = ana_triangular.forward(x_in, q, t, mi, sigma, strategy, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, mi, sigma = ctx.saved_tensors
        if use_ana_cuda_kernels and grad_in.is_cuda:
            grad_out = ana_triangular_cuda.backward(grad_in, x_in, q, t, mi, sigma)
        else:
            grad_out = ana_triangular.backward(grad_in, x_in, q, t, mi, sigma)
        return grad_out, None, None, None, None, None, None


# normal noise

class ANANormal(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and normal noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, mi, sigma, strategy, training):
        ctx.save_for_backward(x_in, q, t, mi, sigma)
        if use_ana_cuda_kernels and x_in.is_cuda:
            x_out = ana_normal_cuda.forward(x_in, q, t, mi, sigma, strategy, torch.Tensor([training]).to(sigma))
        else:
            x_out = ana_normal.forward(x_in, q, t, mi, sigma, strategy, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, mi, sigma = ctx.saved_tensors
        if use_ana_cuda_kernels and grad_in.is_cuda:
            grad_out = ana_normal_cuda.backward(grad_in, x_in, q, t, mi, sigma)
        else:
            grad_out = ana_normal.backward(grad_in, x_in, q, t, mi, sigma)
        return grad_out, None, None, None, None, None, None


# logistic noise

class ANALogistic(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and logistic noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, mi, sigma, strategy, training):
        ctx.save_for_backward(x_in, q, t, mi, sigma)
        if use_ana_cuda_kernels and x_in.is_cuda:
            x_out = ana_logistic_cuda.forward(x_in, q, t, mi, sigma, strategy, torch.Tensor([training]).to(sigma))
        else:
            x_out = ana_logistic.forward(x_in, q, t, mi, sigma, strategy, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, mi, sigma = ctx.saved_tensors
        if use_ana_cuda_kernels and grad_in.is_cuda:
            grad_out = ana_logistic_cuda.backward(grad_in, x_in, q, t, mi, sigma)
        else:
            grad_out = ana_logistic.backward(grad_in, x_in, q, t, mi, sigma)
        return grad_out, None, None, None, None, None, None
