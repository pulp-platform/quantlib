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
import warnings
from typing import NamedTuple, Dict, Union, Any

from quantlib.utils import quantlib_err_header


# -- CANONICAL DATA STRUCTURE -- #

DEFAULT_DTYPE = torch.float32


class ShapeAndDType(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype = DEFAULT_DTYPE


class InputShapesAndDTypes(OrderedDict):

    def __setitem__(self, target: str, shapeanddtype: ShapeAndDType):

        if not isinstance(target, str):  # should be a valid `fx.Node` target
            raise TypeError
        if not isinstance(shapeanddtype, ShapeAndDType):
            raise TypeError

        super(InputShapesAndDTypes, self).__setitem__(target, shapeanddtype)


# -- CANONICALISATION FLOW -- #

InputShapesAndDTypesSpecType = Union[InputShapesAndDTypes, Dict[str, Dict[str, Any]]]


def resolve_inputshapesanddtypes_inputshapesanddtypesspec(inputshapesanddtypesspec: InputShapesAndDTypes) -> InputShapesAndDTypes:
    return inputshapesanddtypesspec


def resolve_dict_inputshapesanddtypesspec(inputshapesanddtypesspec: Dict[str, Dict[str, Any]]) -> InputShapesAndDTypes:

    # each input description should contain the following keys ...
    mandatory_keys = {'shape'}
    # ... and optionally the following ones
    optional_keys = {'dtype'}

    inputshapesanddtype = InputShapesAndDTypes()

    # canonicalise each component
    for target, spec in inputshapesanddtypesspec.items():

        if not (isinstance(spec, dict) and all(isinstance(key, str) for key in spec.keys())):
            raise TypeError

        defined_keys = set(spec.keys())
        unknown_keys = defined_keys.difference(mandatory_keys | optional_keys)
        if len(mandatory_keys.intersection(defined_keys)) == 0:
            raise ValueError
        if len(unknown_keys) > 0:
            warnings.warn('')  # unexpected keys

        # update the canonical data structure
        if optional_keys.issubset(defined_keys):
            shapeanddtype = ShapeAndDType(shape=spec['shape'], dtype=spec['dtype'])
        else:
            shapeanddtype = ShapeAndDType(shape=spec['shape'])

        inputshapesanddtype[target] = shapeanddtype

    # return canonical data structure
    return inputshapesanddtype


InputShapesAndDTypesSolvers = Enum('InputShapesAndDTypesSolvers',
                                   [
                                       ('INPUTSHAPESANDDTYPES', resolve_inputshapesanddtypes_inputshapesanddtypesspec),
                                       ('DICT',                 resolve_dict_inputshapesanddtypesspec),
                                   ])


def resolve_inputshapesanddtypesspec(inputshapesanddtypesspec: InputShapesAndDTypesSpecType) -> InputShapesAndDTypes:

    inputshapesanddtypesspec_class = inputshapesanddtypesspec.__class__.__name__
    try:
        solver = getattr(InputShapesAndDTypesSolvers, inputshapesanddtypesspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qdescription = solver(inputshapesanddtypesspec)
        return qdescription
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported InputShapesAndDTypes specification type: {inputshapesanddtypesspec_class}.")
