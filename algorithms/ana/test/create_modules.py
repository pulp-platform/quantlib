import torch
import torch.nn as nn

import quantlib.algorithms.ana as ana

from typing import Dict, Tuple
from typing import Union


class ModuleFactory(object):

    def __init__(self):
        pass

    def get_module(self,
                   class_name: str,
                   training: bool,
                   device: torch.device,
                   input_size: Tuple[int]) -> [nn.Module, Tuple]:

        raise NotImplementedError

    def get_network(self,
                    linear_or_conv2d: str,
                    training: bool,
                    device: torch.device,
                    input_size: Union[Tuple[int, int], Tuple[int, int, int, int]]) -> Tuple[nn.Module, Union[Tuple[int, int], Tuple[int, int, int, int]]]:

        if linear_or_conv2d == 'linear':

            linear1, linear1_size = self.get_module('linear', training, device, input_size)
            batchnorm1 = nn.BatchNorm1d(linear1_size[-1])
            act1, act1_size = self.get_module('activation', training, device, linear1_size)
            linear2, linear2_size = self.get_module('linear', training, device, act1_size)

            network = nn.Sequential(linear1, batchnorm1, act1, linear2)

            output_size = linear2_size

        elif linear_or_conv2d == 'conv2d':

            conv1, conv1_size = self.get_module('conv2d', training, device, input_size)
            batchnorm1 = nn.BatchNorm2d(conv1_size[1])
            act1, act1_size = self.get_module('activation', training, device, conv1_size)
            conv2, conv2_size = self.get_module('conv2d', training, device, act1_size)

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


class FloatModuleFactory(ModuleFactory):

    def __init__(self):
        super().__init__()

    def get_module(self,
                   class_name: str,
                   training: bool,
                   device: torch.device,
                   input_size: Tuple) -> Tuple[nn.Module, Tuple]:

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

    def __init__(self,
                 quantizer_spec: Dict,
                 noise_type: str,
                 mi:         float,
                 sigma:      float,
                 strategy:   int):

        self._quantizer_spec = quantizer_spec
        self._noise_type = noise_type
        self._mi         = mi
        self._sigma      = sigma
        self._strategy   = strategy

    def get_module(self,
                   class_name: str,
                   training: bool,
                   device: torch.device,
                   input_size: Union[Tuple[int, int], Tuple[int, int, int, int]]) -> [nn.Module, Union[Tuple[int, int], Tuple[int, int, int, int]]]:

        # build module object
        if class_name == 'activation':
            assert len(input_size) in {2, 4}
            class_      = getattr(ana, 'ANAActivation')
            module      = class_(self._quantizer_spec, self._noise_type, self._strategy)
            output_size = input_size

        elif class_name == 'linear':
            assert len(input_size) == 2
            B, N = input_size
            class_ = getattr(ana, 'ANALinear')
            module = class_(self._quantizer_spec, self._noise_type, self._strategy,
                            in_features=N, out_features=N, bias=False)
            output_size = (B, N)

        elif class_name == 'conv2d':
            assert len(input_size) == 4
            B, Nin, Hin, Win = input_size
            Nout = Nin * 2
            K    = 3
            S    = 2
            P    = 1
            class_ = getattr(ana, 'ANAConv2d')
            module = class_(self._quantizer_spec, self._noise_type, self._strategy,
                            in_channels=Nin, out_channels=Nout,
                            kernel_size=K, stride=S, padding=P,
                            bias=False)
            output_size = (B, Nout, Hin // 2, Win // 2)

        else:
            raise ValueError

        # set noise
        module.set_noise(self._mi, self._sigma)

        # set training flag
        if training:
            module.train()
        else:
            module.valid()

        # maybe cast to GPU
        module = module.to(device=device)

        return module, output_size
