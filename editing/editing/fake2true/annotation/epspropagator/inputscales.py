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
