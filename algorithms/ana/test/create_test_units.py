from enum import IntEnum
import torch
import torch.nn as nn

from .create_tensors import BatchSize, InputSize, LinearSize, Conv2dChannels, Conv2dSpatialSize, Kind, Range
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

    def __init__(self, quantizer_spec: Dict):
        self._quantizer_spec = quantizer_spec

    @property
    def range(self) -> Range:
        # TODO: since the quantizer specification is fixed, I can compute the positions of the thresholds and generate inputs that span the whole domain
        # TODO: this function is tightly coupled with the `quantizer_spec` "resolution" performed by the basic `ANAModule` class

        # compute quantization levels
        quant_levels = torch.arange(0, 2 ** self._quantizer_spec['nbits']).to(dtype=torch.float32)
        if self._quantizer_spec['signed']:
            quant_levels = quant_levels - 2 ** (self._quantizer_spec['nbits'] - 1)
            if self._quantizer_spec['balanced']:
                quant_levels = quant_levels[1:]

        # compute thresholds
        thresholds   = quant_levels[:-1] + .5
        max_bin_size = torch.max(thresholds[1:] - thresholds[:-1]).item()
        lower_bound  = torch.min(thresholds).item() - max_bin_size / 2
        upper_bound  = torch.max(thresholds).item() + max_bin_size / 2

        return Range(lower_bound, upper_bound)

    def get_ana_module_factory(self,
                               noise_type: str,
                               mi: float,
                               sigma: float,
                               strategy: int) -> ANAModuleFactory:
        return ANAModuleFactory(self._quantizer_spec, noise_type, mi, sigma, strategy)


class FunctionalEquivalenceUnitGenerator(object):

    def __init__(self, factory_generator: ANAModuleFactoriesGenerator):

        assert torch.cuda.is_available()  # it does not make sense to perform a functional comparison between two clones of the same thing (i.e., CPU and GPU versions)
        self._cpu_device = torch.device('cpu')
        self._gpu_device = torch.device(torch.cuda.current_device())

        self._factory_generator = factory_generator

    def get_test_unit(self,
                      noise_type: str,
                      mi: float,
                      sigma: float,
                      strategy: int,
                      batch_size: BatchSize,
                      input_size: InputSize,
                      test_module: TestModule) -> Tuple[Tuple[TensorGenerator, nn.Module, TensorGenerator], Tuple[TensorGenerator, nn.Module, TensorGenerator]]:

        factory = self._factory_generator.get_ana_module_factory(noise_type, mi, sigma, strategy)

        # create input generators
        batch_size = batch_size.value
        if (test_module == TestModule.ACTIVATIONLINEAR) or (test_module == TestModule.LINEAR):
            input_size_2_linear_size = {
                InputSize.SMALL:  LinearSize.SMALL.value,
                InputSize.NORMAL: LinearSize.NORMAL.value,
                InputSize.LARGE:  LinearSize.LARGE.value
            }
            n_channels = input_size_2_linear_size[input_size]
            x_gen_cpu = LinearTensorGenerator(self._cpu_device, batch_size, n_channels, Kind.LINSPACE, self._factory_generator.range)
            x_gen_gpu = LinearTensorGenerator(self._gpu_device, batch_size, n_channels, Kind.LINSPACE, self._factory_generator.range)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D):
            input_size_2_conv2d_size = {
                InputSize.SMALL: (Conv2dChannels.SMALL.value, Conv2dSpatialSize.SMALL.value),
                InputSize.NORMAL: (Conv2dChannels.NORMAL.value, Conv2dSpatialSize.NORMAL.value),
                InputSize.LARGE: (Conv2dChannels.LARGE.value, Conv2dSpatialSize.LARGE.value)
            }
            n_channels, spatial_size = input_size_2_conv2d_size[input_size]
            x_gen_cpu = Conv2dTensorGenerator(self._cpu_device, batch_size, n_channels, spatial_size, Kind.LINSPACE, self._factory_generator.range)
            x_gen_gpu = Conv2dTensorGenerator(self._gpu_device, batch_size, n_channels, spatial_size, Kind.LINSPACE, self._factory_generator.range)
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
            grad_gen_cpu = LinearTensorGenerator(self._cpu_device, *output_size_cpu, Kind.ONES)
            grad_gen_gpu = LinearTensorGenerator(self._gpu_device, *output_size_gpu, Kind.ONES)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D):
            grad_gen_cpu = Conv2dTensorGenerator(self._cpu_device, *output_size_cpu[:-1], Kind.ONES)
            grad_gen_gpu = Conv2dTensorGenerator(self._gpu_device, *output_size_gpu[:-1], Kind.ONES)
        else:
            raise ValueError

        return (x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu)


class ProfilingUnitGenerator(object):

    def __init__(self,
                 factory: ModuleFactory,
                 device: torch.device):

        self._factory = factory
        self._device  = device

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
            x_gen = LinearTensorGenerator(self._device, batch_size, n_channels, Kind.RANDN)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D) or (test_module == TestModule.CONV2DNETWORK):
            input_size_2_conv2d_size = {
                InputSize.SMALL:  (Conv2dChannels.SMALL.value,  Conv2dSpatialSize.SMALL.value),
                InputSize.NORMAL: (Conv2dChannels.NORMAL.value, Conv2dSpatialSize.NORMAL.value),
                InputSize.LARGE:  (Conv2dChannels.LARGE.value,  Conv2dSpatialSize.LARGE.value)
            }
            n_channels, spatial_size = input_size_2_conv2d_size[input_size]
            x_gen = Conv2dTensorGenerator(self._device, batch_size, n_channels, spatial_size, Kind.RANDN)
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
            grad_gen = LinearTensorGenerator(self._device, *output_size, Kind.ONES)
        elif (test_module == TestModule.ACTIVATIONCONV2D) or (test_module == TestModule.CONV2D) or (test_module == TestModule.CONV2DNETWORK):
            grad_gen = Conv2dTensorGenerator(self._device, *output_size[:-1], Kind.ONES)
        else:
            raise ValueError

        return x_gen, module, grad_gen
