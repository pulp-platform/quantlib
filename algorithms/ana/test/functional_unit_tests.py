# 
# functional_unit_tests.py
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
import numpy as np
import matplotlib.pyplot as plt

import quantlib.algorithms.ana as ana

import torch.nn as nn
from .create_tensors import TensorGenerator


def compare_tensors(t1:        torch.Tensor,
                    t2:        torch.Tensor,
                    tolerance: float) -> bool:
    """Determine whether two tensors are equivalent.

    The tensors must contain numeric components, so that it is possible to
    perform subtraction and to define a total order relationship. The tensors
    are considered equivalent if all their components differ by less than a
    given threshold. Those components whose difference surpasses the threshold
    are printed to screen.

    This function is meant to be a helper function for
    `numerical_equivalence`.
    """

    assert t1.shape == t2.shape

    equivalent = torch.abs(t2 - t1) <= tolerance
    if torch.all(equivalent):
        are_equivalent = True
    else:
        are_equivalent = False
        for coords in torch.logical_not(equivalent).nonzero():
            coords = tuple([c.item() for c in coords])  # https://stackoverflow.com/a/52092886
            print("Difference at position {}: t1[{}] = {}, t2[{}] = {}.".format(coords, coords, t1[coords], coords, t2[coords]))

    return are_equivalent


def numerical_equivalence(x_gen_cpu:    TensorGenerator,
                          module_cpu:   nn.Module,
                          grad_gen_cpu: TensorGenerator,
                          x_gen_gpu:    TensorGenerator,
                          module_gpu:   nn.Module,
                          grad_gen_gpu: TensorGenerator,
                          tolerance:    float = 1e-8) -> bool:
    """Compute whether the CPU and GPU versions of ANA modules are equivalent."""

    assert type(module_cpu) == type(module_gpu)
    assert module_cpu.strategy == module_gpu.strategy.cpu()
    assert module_cpu.strategy != 2  # automatically comparing numerical functional equivalence for non-deterministic functions is complicated;
                                     # for the moment, I leave this responsibility to humans: see the `visual_equivalence` function

    if hasattr(module_cpu, 'weight'):  # `ANALinear` and `ANAConv2d` modules
        assert (module_cpu.bias is None) and (module_gpu.bias is None)
        has_weight = True
    else:  # `ANAActivation` modules
        has_weight = False

    # clone parameters from CPU to GPU version; check that `bias` is `None`
    if has_weight:
        module_gpu.weight.data = module_cpu.weight.data.to(module_gpu.weight.data)

    # process CPU arrays
    x_cpu  = next(x_gen_cpu)
    x_cpu.requires_grad = True
    y_cpu  = module_cpu(x_cpu)
    yg_cpu = next(grad_gen_cpu)
    y_cpu.backward(yg_cpu)

    # process GPU arrays
    x_gpu  = next(x_gen_gpu)
    x_gpu.requires_grad = True
    y_gpu  = module_gpu(x_gpu)
    yg_gpu = next(grad_gen_gpu)
    y_gpu.backward(yg_gpu)

    # compare arrays
    # input-related
    dx     = compare_tensors(x_cpu, x_gpu.cpu(), 0.0)  # should be deterministic
    dxg    = compare_tensors(x_cpu.grad, x_gpu.grad.cpu(), tolerance)
    x_test = dx and dxg
    # output-related
    dy     = compare_tensors(y_cpu, y_gpu.cpu(), tolerance)
    dyg    = compare_tensors(yg_cpu, yg_gpu.cpu(), tolerance)  # should be deterministic
    y_test = dy and dyg
    # parameter-related
    if has_weight:
        dw     = compare_tensors(module_cpu.weight_maybe_quant, module_gpu.weight_maybe_quant.cpu(), tolerance)
        dwg    = compare_tensors(module_cpu.weight.grad, module_gpu.weight.grad.cpu(), tolerance)
        w_test = dw and dwg
    else:
        w_test = True  # no test to make, so this should not result in failure

    return x_test and y_test and w_test


def scatterplot2d(x:   torch.Tensor,
                  y:   torch.Tensor,
                  eps: torch.Tensor) -> None:
    """Show a sample of a (deterministic) function.

    Functions implemented on a digital computer can only be evaluated on a
    finite (although large) collection of input points. Therefore, a scatter
    plot is more precise than a line plot, which would "interpolate" the value
    of the function on points comprised in-between points where the function
    has actually been computed.

    This function is meant to be a helper function for `visual_equivalence`.
    """

    plt.scatter(x, y, s=4, marker='.')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min() - eps / 2, y.max() + eps / 2)


def distribution2d(x_all:        torch.Tensor,
                   y_all:        torch.Tensor,
                   quant_levels: torch.Tensor,
                   eps:          torch.Tensor) -> None:
    """Show the joint distribution of two variables `x` and `y` as a heatmap.

    When the relationship between two sets `X` and `Y` is non-deterministic
    (i.e., it does not satisfy the definition of function), it can still be
    visualised by plotting the empirical distribution obtained by sampling
    from it multiple `times.

    This function is meant to be a helper function for `visual_equivalence`.
    """

    x = x_all[0]
    assert torch.all(x == x_all)  # https://discuss.pytorch.org/t/check-row-equality-of-tensor/41395/2
    assert torch.all(x == torch.sort(x)[0])  # I assume that the inputs are generated by a `TensorGenerator` with `self._kind == Kind.LINSPACE`

    x_edges = torch.hstack([x, x.max() + torch.max(x[1:] - x[:-1])])
    y_edges = torch.hstack([torch.min(quant_levels) - eps / 2, (quant_levels[1:] + quant_levels[:-1]) / 2, torch.max(quant_levels) + eps / 2])

    counts, x_edges, y_edges = np.histogram2d(x_all.flatten().numpy(),
                                              y_all.flatten().numpy(),
                                              bins=(x_edges.numpy(), y_edges.numpy()))

    plt.imshow(np.transpose(counts),
               interpolation='nearest',
               origin='lower',
               extent=[y_edges[0], y_edges[-1], x_edges[0], x_edges[-1]])


def distribution1d(t: torch.Tensor) -> None:
    """Show the one-dimensional distribution of an array's components.

    When the objects stored in the components of an array are numbers, it is
    possible to interpret the array as a finite sample from a probability
    distribution on the numeric set. The natural way to represent such sample
    distributions is a histogram.

    This function is meant to be a helper function for `visual_equivalence`.
    """
    edges = torch.linspace(t.min(), t.max(), steps=t.numel() // 10)
    counts, edges = np.histogram(t, edges)
    plt.hist(edges[:-1], edges, weights=counts)


def visual_equivalence(x_gen_cpu:    TensorGenerator,
                       module_cpu:   nn.Module,
                       grad_gen_cpu: TensorGenerator,
                       x_gen_gpu:    TensorGenerator,
                       module_gpu:   nn.Module,
                       grad_gen_gpu: TensorGenerator,
                       N:            int = 1000) -> None:
    """Provide information that humans can use to decide whether the CPU and
    GPU implementations are equivalent.
    """

    assert type(module_cpu) == type(module_gpu)
    assert module_cpu.strategy == module_gpu.strategy.cpu()

    if isinstance(module_cpu, ana.ANAActivation):

        fig = plt.figure()

        # process CPU arrays
        if module_cpu.strategy == 2:

            assert N > 0  # at least one iteration (although this is unlikely to yield statistically significant results)

            # process CPU arrays
            for n in range(0, N):
                x_cpu = next(x_gen_cpu)
                x_cpu.requires_grad = True
                y_cpu = module_cpu(x_cpu)
                try:
                    x_cpu_all = torch.vstack([x_cpu_all, x_cpu.flatten()])
                    y_cpu_all = torch.vstack([y_cpu_all, y_cpu.flatten()])
                except UnboundLocalError:
                    x_cpu_all = x_cpu.flatten()
                    y_cpu_all = y_cpu.flatten()
            yg_cpu = next(grad_gen_cpu)
            y_cpu.backward(yg_cpu)

            ax = fig.add_subplot(2, 2, 1, title='Forward (CPU)')
            distribution2d(x_cpu_all.detach(), y_cpu_all.detach(), module_cpu.quant_levels, module_cpu.eps)
            ax = fig.add_subplot(2, 2, 3, title='Backward (CPU)')
            scatterplot2d(x_cpu.detach(), x_cpu.grad.detach(), module_cpu.eps)

            # process GPU arrays
            for n in range(0, N):
                x_gpu = next(x_gen_gpu)
                x_gpu.requires_grad = True
                y_gpu = module_gpu(x_gpu)
                try:
                    x_gpu_all = torch.vstack([x_gpu_all, x_gpu])
                    y_gpu_all = torch.vstack([y_gpu_all, y_gpu])
                except UnboundLocalError:
                    x_gpu_all = x_gpu
                    y_gpu_all = y_gpu
            yg_gpu = next(grad_gen_gpu)
            y_gpu.backward(yg_gpu)

            ax = fig.add_subplot(2, 2, 2, title='Forward (GPU)')
            distribution2d(x_gpu_all.detach(), y_gpu_all.detach(), module_gpu.quant_levels, module_gpu.eps)
            ax = fig.add_subplot(2, 2, 4, title='Backward (GPU)')
            scatterplot2d(x_gpu.detach(), x_gpu.grad.detach(), module_gpu.eps)

        elif (module_cpu.strategy == 0) or (module_cpu.strategy == 1):

            # process CPU arrays
            x_cpu = next(x_gen_cpu)
            x_cpu.requires_grad = True
            y_cpu = module_gpu(x_cpu)
            yg_cpu = next(grad_gen_cpu)
            y_cpu.backward(yg_cpu)

            ax = fig.add_subplot(2, 2, 1, title='Forward (CPU)')
            scatterplot2d(x_cpu.detach(), y_cpu.detach(), module_cpu.eps)
            ax = fig.add_subplot(2, 2, 3, title='Backward (CPU)')
            scatterplot2d(x_cpu.detach(), x_cpu.grad.detach(), module_cpu.eps)

            # process GPU arrays
            x_gpu = next(x_gen_gpu)
            x_gpu.requires_grad = True
            y_gpu = module_gpu(x_gpu)
            yg_gpu = next(grad_gen_gpu)
            y_gpu.backward(yg_gpu)

            ax = fig.add_subplot(2, 2, 2, title='Forward (GPU)')
            scatterplot2d(x_gpu.detach(), y_gpu.detach(), module_gpu.eps)
            ax = fig.add_subplot(2, 2, 4, title='Backward (GPU)')
            scatterplot2d(x_gpu.detach(), x_gpu.grad.detach(), module_gpu.eps)

        else:
            raise ValueError  # invalid strategy

    elif isinstance(module_cpu, ana.ANALinear) or isinstance(module_cpu, ana.ANAConv2d):

        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1, title='Initial distribution (CPU)')
        distribution1d(module_cpu.weight.data.detach())

        ax = fig.add_subplot(1, 2, 2, title='Initial distribution (GPU)')
        distribution1d(module_gpu.weight.data.detach())

    else:
        raise ValueError  # I only support equivalence unit tests for `ANAActivation`, `ANALinear` and `ANAConv2d`

    plt.show()

