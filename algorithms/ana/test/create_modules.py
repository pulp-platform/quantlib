# 
# create_modules.py
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
import torch.nn as nn

from .create_tensors import Range
import quantlib.algorithms.ana as ana

from typing import Dict, Tuple
from typing import Union


class ModuleFactory(object):

    def __init__(self):
        pass

    def get_module(self,
                   class_name: str,
                   training:   bool,
                   device:     torch.device,
                   input_size: Union[Tuple[int, int], Tuple[int, int, int, int]],
                   **kwargs) -> [nn.Module, Union[Tuple[int, int], Tuple[int, int, int, int]]]:

        raise NotImplementedError

    def get_network(self,
                    linear_or_conv2d: str,
                    training:         bool,
                    device:           torch.device,
                    input_size:       Union[Tuple[int, int], Tuple[int, int, int, int]],
                    **kwargs) -> Tuple[nn.Module, Union[Tuple[int, int], Tuple[int, int, int, int]]]:

        if linear_or_conv2d == 'linear':

            linear1, linear1_size = self.get_module('linear', training, device, input_size, **kwargs)
            batchnorm1            = nn.BatchNorm1d(linear1_size[-1])
            act1, act1_size       = self.get_module('activation', training, device, linear1_size, **kwargs)
            linear2, linear2_size = self.get_module('linear', training, device, act1_size, **kwargs)

            network = nn.Sequential(linear1, batchnorm1, act1, linear2)

            output_size = linear2_size

        elif linear_or_conv2d == 'conv2d':

            conv1, conv1_size = self.get_module('conv2d', training, device, input_size, **kwargs)
            batchnorm1        = nn.BatchNorm2d(conv1_size[1])
            act1, act1_size   = self.get_module('activation', training, device, conv1_size, **kwargs)
            conv2, conv2_size = self.get_module('conv2d', training, device, act1_size, **kwargs)

            network = nn.Sequential(conv1, batchnorm1, act1, conv2)

            output_size = conv2_size

        else:
            raise ValueError

        if training:
            network.train()
        else:
            network.valid()
        network = network.to(device=device)

        return network, output_size


class FloatingPointModuleFactory(ModuleFactory):

    def __init__(self):
        super().__init__()

    def get_module(self,
                   class_name: str,
                   training:   bool,
                   device:     torch.device,
                   input_size: Tuple,
                   **kwargs) -> Tuple[nn.Module, Tuple]:

        # build module object
        if class_name == 'activation':
            assert len(input_size) in {2, 4}
            class_ = getattr(nn, 'ReLU')
            module = class_(inplace=True)
            output_size = input_size

        elif class_name == 'linear':
            assert len(input_size) == 2
            B, N = input_size
            class_ = getattr(nn, 'Linear')
            module = class_(in_features=N, out_features=N, bias=False)
            output_size = (B, N)

        elif class_name == 'conv2d':
            assert len(input_size) == 4
            B, Nin, Hin, Win = input_size
            Nout = Nin * 2
            K = 3
            S = 2
            P = 1
            class_ = getattr(nn, 'Conv2d')
            module = class_(in_channels=Nin, out_channels=Nout,
                            kernel_size=K, stride=S, padding=P,
                            bias=False)
            output_size = (B, Nout, Hin // 2, Win // 2)

        else:
            raise ValueError

        # set training flag
        if training:
            module.train()
        else:
            module.valid()

        # maybe cast to GPU
        module = module.to(device=device)

        return module, output_size


class ANAModuleFactory(ModuleFactory):

    def __init__(self, quantizer_spec: Dict):
        super().__init__()
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

    def get_module(self,
                   class_name: str,
                   training:   bool,
                   device:     torch.device,
                   input_size: Union[Tuple[int, int], Tuple[int, int, int, int]],
                   noise_type: Union[str, None]   = None,
                   mi:         Union[float, None] = None,
                   sigma:      Union[float, None] = None,
                   strategy:   Union[str, None]   = None) -> [nn.Module, Union[Tuple[int, int], Tuple[int, int, int, int]]]:

        # build module object
        if class_name == 'activation':
            assert len(input_size) in {2, 4}
            class_      = getattr(ana, 'ANAActivation')
            module      = class_(self._quantizer_spec, noise_type, strategy)
            output_size = input_size

        elif class_name == 'linear':
            assert len(input_size) == 2
            B, N = input_size
            class_ = getattr(ana, 'ANALinear')
            module = class_(self._quantizer_spec, noise_type, strategy,
                            in_features=N, out_features=N, bias=False)
            output_size = (B, N)

        elif class_name == 'conv2d':
            assert len(input_size) == 4
            B, Nin, Hin, Win = input_size
            Nout = Nin * 2
            K    = 3  # \
            S    = 2  # -}---> with this kernel hyper-parameters the spatial dimensions of the output are halved with respect to those of the input
            P    = 1  # /
            class_ = getattr(ana, 'ANAConv2d')
            module = class_(self._quantizer_spec, noise_type, strategy,
                            in_channels=Nin, out_channels=Nout,
                            kernel_size=K, stride=S, padding=P,
                            bias=False)
            output_size = (B, Nout, Hin // 2, Win // 2)

        else:
            raise ValueError

        # set noise
        module.set_noise(mi, sigma)

        # set training flag
        if training:
            module.train()
        else:
            module.valid()

        # maybe cast to GPU
        module = module.to(device=device)

        return module, output_size


def create_quantizer_spec(nbits:    int,
                          signed:   bool,
                          balanced: bool,
                          eps:      float):

    quantizer_spec = {
        'nbits':    nbits,
        'signed':   signed,
        'balanced': balanced,
        'eps':      eps
    }

    return quantizer_spec

