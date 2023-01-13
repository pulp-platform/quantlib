# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
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

from collections import OrderedDict
from enum import Enum
import torch
from typing import Dict, Union

from quantlib.utils import quantlib_err_header


# -- CANONICAL DATA STRUCTURE -- #

class InputScales(OrderedDict):

    def __setitem__(self, target: str, scale: torch.Tensor):

        if not isinstance(target, str):  # should be a valid `fx.Node` target
            raise TypeError
        if not isinstance(scale, torch.Tensor):
            raise TypeError

        super(InputScales, self).__setitem__(target, scale)


# -- CANONICALISATION FLOW -- #

InputScalesSpecType = Union[InputScales, Dict[str, Union[torch.Tensor, float]]]


def resolve_inputscales_inputscalesspec(inputscalesspec: InputScales) -> InputScales:
    return inputscalesspec


def resolve_dict_inputscalesspec(inputscalesspec: Dict[str, Union[torch.Tensor, float]]) -> InputScales:

    # validate input types
    if not all(isinstance(target, str) for target in inputscalesspec.keys()):
        raise TypeError
    if not all(isinstance(scale, (torch.Tensor, float)) for scale in inputscalesspec.values()):
        raise TypeError

    # create canonical data structure
    inputscales = InputScales()
    # parse the input
    for target, scale in inputscalesspec.items():

        if isinstance(scale, float):  # canonicalise it to `torch.Tensor`
            scale = torch.Tensor([scale])

        inputscales[target] = scale

    return inputscales


InputScalesSolvers = Enum('InputScalesSolvers',
                          [
                              ('INPUTSCALES', resolve_inputscales_inputscalesspec),
                              ('DICT',        resolve_dict_inputscalesspec),
                          ])


def resolve_inputscalesspec(inputscalesspec: InputScalesSpecType) -> InputScales:

    inputscalesspec_class = inputscalesspec.__class__.__name__
    try:
        solver = getattr(InputScalesSolvers, inputscalesspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qdescription = solver(inputscalesspec)
        return qdescription
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported InputScales specification type: {inputscalesspec_class}.")
