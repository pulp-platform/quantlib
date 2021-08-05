from enum import IntEnum
import torch
import torch.nn as nn

from .create_tensors import Kind, BatchSize, InputSize, LinearSize, Conv2dChannels, Conv2dSpatialSize
from .create_tensors import LinearTensorGenerator, Conv2dTensorGenerator
from .create_modules import ANAModuleFactory

from typing import Tuple, Dict

from .create_tensors import TensorGenerator
from .create_modules import ModuleFactory


class TestModule(IntEnum):
    ACTIVATIONLINEAR = 0
    ACTIVATIONCONV2D = 1
    LINEAR           = 2
    CONV2D           = 3
    LINEARNETWORK    = 4
    CONV2DNETWORK    = 5


def create_quantizer_spec(nbits: int,
                          signed: bool,
                          balanced: bool,
                          eps: float):

    quantizer_spec = {
        'nbits':    nbits,
        'signed':   signed,
        'balanced': balanced,
        'eps':      eps
    }

    return quantizer_spec


class ANAModuleFactoriesGenerator(object):

    def __init__(self,
                 quantizer_spec: Dict,
                 mi: float,
                 sigma: float):

        self._quantizer_spec = quantizer_spec
        self._mi    = mi
        self._sigma = sigma

    def get_ana_module_factory(self,
                               noise_type: str,
                               strategy: int) -> ANAModuleFactory:
        return ANAModuleFactory(self._quantizer_spec, noise_type, self._mi, self._sigma, strategy)


class FunctionalEquivalenceUnitGenerator(object):

    def __init__(self, factory: ANAModuleFactoriesGenerator):

        assert torch.cuda.is_available()  # it does not make sense to perform a functional comparison between two clones of the same thing (i.e., CPU and GPU versions)
        self._cpu_device = torch.device('cpu')
        self._gpu_device = torch.device(torch.cuda.current_device())

        self._factory = factory

    def get_test_unit(self,
                      noise_type: str,
                      strategy: int,
                      batch_size: BatchSize,
                      input_size: InputSize,
                      test_module: TestModule) -> Tuple[Tuple[TensorGenerator, nn.Module, TensorGenerator], Tuple[TensorGenerator, nn.Module, TensorGenerator]]:

        factory = self._factory.get_ana_module_factory(noise_type, strategy)

        # create input generators
        batch_size = batch_size.value
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.LINEAR):
            input_size_2_linear_size = {
                InputSize.SMALL:  LinearSize.SMALL.value,
                InputSize.NORMAL: LinearSize.NORMAL.value,
                InputSize.LARGE:  LinearSize.LARGE.value
            }
            n_channels = input_size_2_linear_size[input_size]
            x_gen_cpu = LinearTensorGenerator(self._cpu_device, Kind.LINSPACE, batch_size, n_channels)
            x_gen_gpu = LinearTensorGenerator(self._gpu_device, Kind.LINSPACE, batch_size, n_channels)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D):
            input_size_2_conv2d_size = {
                InputSize.SMALL: (Conv2dChannels.SMALL.value, Conv2dSpatialSize.SMALL.value),
                InputSize.NORMAL: (Conv2dChannels.NORMAL.value, Conv2dSpatialSize.NORMAL.value),
                InputSize.LARGE: (Conv2dChannels.LARGE.value, Conv2dSpatialSize.LARGE.value)
            }
            n_channels, spatial_size = input_size_2_conv2d_size[input_size]
            x_gen_cpu = Conv2dTensorGenerator(self._cpu_device, Kind.LINSPACE, batch_size, n_channels, spatial_size)
            x_gen_gpu = Conv2dTensorGenerator(self._gpu_device, Kind.LINSPACE, batch_size, n_channels, spatial_size)
        else:
            raise ValueError

        # create modules
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.ACTIVATIONCONV2D):
            module_cpu, output_size_cpu = factory.get_module('activation', True, self._cpu_device, x_gen_cpu.size)
            module_gpu, output_size_gpu = factory.get_module('activation', True, self._gpu_device, x_gen_gpu.size)
        elif test_module == TestModule.LINEAR:
            module_cpu, output_size_cpu = factory.get_module('linear', True, self._cpu_device, x_gen_cpu.size)
            module_gpu, output_size_gpu = factory.get_module('linear', True, self._gpu_device, x_gen_gpu.size)
        elif test_module == TestModule.CONV2D:
            module_cpu, output_size_cpu = factory.get_module('conv2d', True, self._cpu_device, x_gen_cpu.size)
            module_gpu, output_size_gpu = factory.get_module('conv2d', True, self._gpu_device, x_gen_gpu.size)
        else:
            raise ValueError

        # create gradient generators
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.LINEAR):
            grad_gen_cpu = LinearTensorGenerator(self._cpu_device, Kind.ONES, *output_size_cpu)
            grad_gen_gpu = LinearTensorGenerator(self._gpu_device, Kind.ONES, *output_size_gpu)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D):
            grad_gen_cpu = Conv2dTensorGenerator(self._cpu_device, Kind.ONES, *output_size_cpu[:-1])
            grad_gen_gpu = Conv2dTensorGenerator(self._gpu_device, Kind.ONES, *output_size_gpu[:-1])
        else:
            raise ValueError

        return (x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu)


class ProfilingUnitGenerator(object):

    def __init__(self,
                 device: torch.device,
                 factory: ModuleFactory):

        self._device  = device
        self._factory = factory

    def get_test_unit(self,
                      batch_size: BatchSize,
                      input_size: InputSize,
                      test_module: TestModule) -> Tuple[TensorGenerator, nn.Module, TensorGenerator]:

        # create input generator
        batch_size = batch_size.value
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.LINEAR) or (test_module == TestModule.LINEARNETWORK):
            input_size_2_linear_size = {
                InputSize.SMALL:  LinearSize.SMALL.value,
                InputSize.NORMAL: LinearSize.NORMAL.value,
                InputSize.LARGE:  LinearSize.LARGE.value
            }
            n_channels = input_size_2_linear_size[input_size]
            x_gen = LinearTensorGenerator(self._device, Kind.RANDN, batch_size, n_channels)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D) or (test_module == TestModule.CONV2DNETWORK):
            input_size_2_conv2d_size = {
                InputSize.SMALL:  (Conv2dChannels.SMALL.value,  Conv2dSpatialSize.SMALL.value),
                InputSize.NORMAL: (Conv2dChannels.NORMAL.value, Conv2dSpatialSize.NORMAL.value),
                InputSize.LARGE:  (Conv2dChannels.LARGE.value,  Conv2dSpatialSize.LARGE.value)
            }
            n_channels, spatial_size = input_size_2_conv2d_size[input_size]
            x_gen = Conv2dTensorGenerator(self._device, Kind.RANDN, batch_size, n_channels, spatial_size)
        else:
            raise ValueError

        # create module and get output size
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.ACTIVATIONCONV2D):
            module, output_size = self._factory.get_module('activation', True, self._device, x_gen.size)
        elif test_module == TestModule.LINEAR:
            module, output_size = self._factory.get_module('linear', True, self._device, x_gen.size)
        elif test_module == TestModule.CONV2D:
            module, output_size = self._factory.get_module('conv2d', True, self._device, x_gen.size)
        elif test_module == TestModule.LINEARNETWORK:
            module, output_size = self._factory.get_network('linear', True, self._device, x_gen.size)
        elif test_module == TestModule.CONV2DNETWORK:
            module, output_size = self._factory.get_network('conv2d', True, self._device, x_gen.size)
        else:
            raise ValueError

        # create gradient generator
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.LINEAR) or (test_module == TestModule.LINEARNETWORK):
            grad_gen = LinearTensorGenerator(self._device, Kind.ONES, *output_size)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D) or (test_module == TestModule.CONV2DNETWORK):
            grad_gen = Conv2dTensorGenerator(self._device, Kind.ONES, *output_size[:-1])
        else:
            raise ValueError

        return x_gen, module, grad_gen
