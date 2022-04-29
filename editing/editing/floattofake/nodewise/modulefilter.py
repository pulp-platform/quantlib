from enum import Enum
import functools
import re
import torch.nn as nn
from typing import Dict, Tuple, Callable, Any, Union, Type

from quantlib.utils import quantlib_err_header


ModuleFilter = Callable[[Dict[str, nn.Module]], Dict[str, nn.Module]]


ModuleFilterSpecType = Union[ModuleFilter, Dict[str, Any], str]


def resolve_modulefilter_modulefilterspec(modulefilterspec: ModuleFilter) -> ModuleFilter:
    return modulefilterspec


def null_filter(name_to_module: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
    result = name_to_module.copy()
    return result


def regex_filter(name_to_module: Dict[str, nn.Module], regex: str) -> Dict[str, nn.Module]:
    result = {n: m for n, m in name_to_module.items() if re.match(regex, n)}
    return result


def types_filter(name_to_module: Dict[str, nn.Module], types: Tuple[Type[nn.Module], ...]) -> Dict[str, nn.Module]:
    result = {n: m for n, m in name_to_module.items() if isinstance(m, types)}
    return result


def inclusive_names_filter(name_to_module: Dict[str, nn.Module], names: Tuple[str, ...]) -> Dict[str, nn.Module]:
    result = {n: m for n, m in name_to_module.items() if n in names}
    return result


def exclusive_names_filter(name_to_module: Dict[str, nn.Module], names: Tuple[str, ...]) -> Dict[str, nn.Module]:
    result = {n: m for n, m in name_to_module.items() if n not in names}
    return result


def compose(*filters: ModuleFilter) -> ModuleFilter:
    return functools.reduce(lambda f, g: lambda n2m: g(f(n2m)), filters)  # https://www.youtube.com/watch?v=ka70COItN40&t=1346s


_STRING_TO_NNMODULECLASS = {
    'nn.Identity':  nn.Identity,
    'nn.ReLU':      nn.ReLU,
    'nn.ReLU6':     nn.ReLU6,
    'nn.LeakyReLU': nn.LeakyReLU,
    'nn.Linear':    nn.Linear,
    'nn.Conv1d':    nn.Conv1d,
    'nn.Conv2d':    nn.Conv2d,
    'nn.Conv3d':    nn.Conv3d,
}


def resolve_dict_modulefilterspec(modulefilterspec: Dict[str, Any]) -> ModuleFilter:

    regexfilter_keys   = {'regex'}
    typesfilter_keys   = {'types'}
    excludefilter_keys = {'exclude'}

    modulefilterspec_keys = set(modulefilterspec.keys())
    unknown_keys = modulefilterspec_keys.difference(regexfilter_keys | typesfilter_keys | excludefilter_keys)
    if len(unknown_keys) != 0:
        raise ValueError(quantlib_err_header() + f"ModuleFilter dictionary specification does not support the following keys: {unknown_keys}.")

    if regexfilter_keys.issubset(modulefilterspec_keys):
        regex = modulefilterspec['regex']
        regex_f = functools.partial(regex_filter, regex=regex)
    else:
        regex_f = null_filter

    if typesfilter_keys.issubset(modulefilterspec_keys):
        types = modulefilterspec['types']
        if not isinstance(types, tuple):  # tuplify
            types = (types,)
        types = tuple(map(lambda t: _STRING_TO_NNMODULECLASS[t], types))
        types_f = functools.partial(types_filter, types=types)
    else:
        types_f = null_filter

    if excludefilter_keys.issubset(modulefilterspec_keys):
        exclude = modulefilterspec['exclude']
        if not isinstance(exclude, tuple):
            exclude = (exclude,)
        exclude_f = functools.partial(exclusive_names_filter, names=exclude)
    else:
        exclude_f = null_filter

    modulefilter = compose(regex_f, types_f, exclude_f)
    return modulefilter


def resolve_str_modulefilterspec(modulefilterspec: str) -> ModuleFilter:
    modulefilter = functools.partial(inclusive_names_filter, names=(modulefilterspec,))
    return modulefilter


ModuleFilterSpecSolvers = Enum('ModuleFilterSpecSolvers',
                               [
                                   ('MODULEFILTER', resolve_modulefilter_modulefilterspec),
                                   ('DICT',         resolve_dict_modulefilterspec),
                                   ('STR',          resolve_str_modulefilterspec),
                               ])


def resolve_modulefilterspec(modulefilterspec: ModuleFilterSpecType) -> ModuleFilter:

    # I apply a strategy pattern to retrieve the correct solver method based on the ModuleFilter specification type
    modulefilterspec_class = modulefilterspec.__class__.__name__
    try:
        solver = getattr(ModuleFilterSpecSolvers, modulefilterspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        modulefilter = solver(modulefilterspec)
        return modulefilter
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported ModuleFilter specification type: {modulefilterspec_class}.")
