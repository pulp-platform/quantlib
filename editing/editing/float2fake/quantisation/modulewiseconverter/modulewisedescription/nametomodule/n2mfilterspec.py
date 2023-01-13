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
import functools
from enum import Enum
from typing import Dict, Union, Any

from .n2mfilter import N2MFilter
from .n2mfilter import null_filter, regex_filter, types_filter, inclusive_names_filter, exclusive_names_filter, compose
from quantlib.algorithms.qmodules import SUPPORTED_FPMODULES
from quantlib.utils import quantlib_err_header


CLASSNAME_TO_NNMODULE = OrderedDict([
    (class_.__name__, class_) for class_ in SUPPORTED_FPMODULES
])


N2MFilterSpecType = Union[N2MFilter, Dict[str, Any], str]


def resolve_n2mfilter_n2mfilterspec(n2mfilterspec: N2MFilter) -> N2MFilter:
    return n2mfilterspec


def resolve_dict_n2mfilterspec(n2mfilterspec: Dict[str, Any]) -> N2MFilter:

    regexfilter_keys   = {'regex'}
    typesfilter_keys   = {'types'}
    excludefilter_keys = {'exclude'}

    n2mfilterspec_keys = set(n2mfilterspec.keys())
    unknown_keys = n2mfilterspec_keys.difference(regexfilter_keys | typesfilter_keys | excludefilter_keys)
    if len(unknown_keys) > 0:
        raise ValueError(quantlib_err_header() + f"N2MFilter dictionary specification does not support the following keys: {unknown_keys}.")

    if regexfilter_keys.issubset(n2mfilterspec_keys):
        regex = n2mfilterspec['regex']
        regex_f = functools.partial(regex_filter, regex=regex)
    else:
        regex_f = null_filter

    if typesfilter_keys.issubset(n2mfilterspec_keys):
        types = n2mfilterspec['types']
        if not isinstance(types, tuple):  # tuplify
            types = (types,)
        types = tuple(map(lambda t: CLASSNAME_TO_NNMODULE[t], types))
        types_f = functools.partial(types_filter, types=types)
    else:
        types_f = null_filter

    if excludefilter_keys.issubset(n2mfilterspec_keys):
        exclude = n2mfilterspec['exclude']
        if not isinstance(exclude, tuple):
            exclude = (exclude,)
        exclude_f = functools.partial(exclusive_names_filter, names=exclude)
    else:
        exclude_f = null_filter

    n2mfilter = compose(regex_f, types_f, exclude_f)

    return n2mfilter


def resolve_str_n2mfilterspec(n2mfilterspec: str) -> N2MFilter:
    n2mfilter = functools.partial(inclusive_names_filter, names=(n2mfilterspec,))
    return n2mfilter


N2MFilterSpecSolvers = Enum('N2MFilterSpecSolvers',
                            [
                                ('MODULEFILTER', resolve_n2mfilter_n2mfilterspec),
                                ('DICT',         resolve_dict_n2mfilterspec),
                                ('STR',          resolve_str_n2mfilterspec),
                            ])


def resolve_n2mfilterspec(n2mfilterspec: N2MFilterSpecType) -> N2MFilter:

    # I apply a strategy pattern to retrieve the correct solver method based on the N2MFilter specification type
    n2mfilterspec_class = n2mfilterspec.__class__.__name__
    try:
        solver = getattr(N2MFilterSpecSolvers, n2mfilterspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        n2mfilter = solver(n2mfilterspec)
        return n2mfilter
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported N2MFilter specification type: {n2mfilterspec_class}.")
