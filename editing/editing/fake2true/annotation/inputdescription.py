from collections import OrderedDict
from enum import Enum
import torch
import warnings
from typing import NamedTuple, Dict, Union, Any

from .shapepropagator.shapeanddtype import DEFAULT_DTYPE
from .epspropagator.propagationrules import UNDEFINED_EPS
from quantlib.utils import quantlib_err_header


class PlaceholderDescription(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    scale: torch.Tensor


class InputDescription(OrderedDict):

    def __setitem__(self, target: str, description: PlaceholderDescription):

        if not isinstance(target, str):
            raise TypeError
        if not isinstance(description, PlaceholderDescription):
            raise TypeError

        super(InputDescription, self).__setitem__(target, description)


InputDescriptionSpecType = Union[InputDescription, Dict[str, Dict[str, Any]]]


def resolve_inputdescription_inputdescriptionspec(inputdescriptionspec: InputDescription) -> InputDescription:
    return inputdescriptionspec


def resolve_dict_inputdescriptionspec(inputdescriptionspec: Dict[str, Dict[str, Any]]) -> InputDescription:

    # validate input type
    if not all(isinstance(target, str) for target in inputdescriptionspec.keys()):
        raise TypeError
    if not all((isinstance(description, dict) and all(isinstance(key, str) for key in description.keys())) for description in inputdescriptionspec.values()):
        raise TypeError

    # parse and canonicalise the description
    inputdescription = InputDescription()

    mandatory_keys = {'shape'}
    optional_kwargs = {
        'dtype': DEFAULT_DTYPE,
        'scale': UNDEFINED_EPS,
    }
    optional_keys = set(optional_kwargs.keys())

    for target, description in inputdescriptionspec.items():

        described_keys = set(description.keys())

        # the user must specify these values for each target
        if not mandatory_keys.issubset(set(described_keys)):
            raise ValueError  # missing key
        placeholder_description = {k: v for k, v in description.items() if (k in mandatory_keys)}

        # we assign default values to the optional arguments, and update them if the user is more diligent
        placeholder_description.update(**optional_kwargs)
        unknown_keys = described_keys.difference(mandatory_keys | optional_keys)
        if len(unknown_keys) > 0:
            warnings.warn('')  # unknown key
            description = {k: v for k, v in description.items() if (k not in unknown_keys)}
        placeholder_description.update(**description)

        # convert to the canonical data structure
        inputdescription[target] = PlaceholderDescription(**placeholder_description)

    return inputdescription


InputDescriptionSolvers = Enum('InputDescriptionSolvers',
                               [
                                   ('INPUTDESCRIPTION', resolve_inputdescription_inputdescriptionspec),
                                   ('DICT',             resolve_dict_inputdescriptionspec),
                               ])


def resolve_inputdescriptionspec(inputdescriptionspec: InputDescriptionSpecType) -> InputDescription:

    inputdescriptionspec_class = inputdescriptionspec.__class__.__name__
    try:
        solver = getattr(InputDescriptionSolvers, inputdescriptionspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qdescription = solver(inputdescriptionspec)
        return qdescription
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported InputDescription specification type: {inputdescriptionspec_class}.")
