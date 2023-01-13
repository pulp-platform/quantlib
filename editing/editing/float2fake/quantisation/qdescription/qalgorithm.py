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

from enum import Enum
from typing import NamedTuple, Tuple, Dict, Union, Optional, Any

from quantlib.algorithms.qalgorithms import ModuleMapping, register
from quantlib.utils import quantlib_err_header


# -- CANONICAL DATA STRUCTURE -- #

KwArgsType = Dict[str, Any]


class QAlgorithm(NamedTuple):
    """Collect information about algorithm-specific ``_QModule`` sub-classes.

    We assume that each PTQ/QAT algorithm supported by QuantLib is implemented
    in the ``quantlib.algorithms.qalgorithms`` namespace. Each such algorithm
    must define a mapping from floating-point ``nn.Module``s to their
    fake-quantised counterparts, which should be sub-classes of ``_QModule``.

    Each of these sub-classes might support algorithm-specific keyword
    arguments that can be passed to their constructor methods.

    """
    mapping: ModuleMapping
    kwargs:  Optional[KwArgsType] = None  # keyword arguments supported by the quantisation algorithm's `_QModule`s


# -- CANONICALISATION FLOW -- #

# First, we enumerate the possible representations of the data structure (both the canonical one and the non-canonical).
QAlgorithmSpecType = Union[QAlgorithm, Tuple[str, KwArgsType], str]


# Then, we define a canonicalisation function ("solver") for each representation.
# Start from the canonical representation (the solver is the identity), ...
def resolve_qalgorithm_qalgorithmspec(qalgorithmspec: QAlgorithm) -> QAlgorithm:
    return qalgorithmspec


# ... then move to the non-canonical ones, from the most specific ...
def resolve_tuple_qalgorithmspec(qalgorithmspec: Tuple[str, Dict[str, Any]]) -> QAlgorithm:

    acronym, kwargs = qalgorithmspec
    try:
        mapping = register[acronym]
    except KeyError:
        raise ValueError

    # TODO: the mapping should also contain information about default keyword
    #       arguments for its `_QModule`s (e.g., default values); we could use
    #       this information for type checking.

    return QAlgorithm(mapping=mapping, kwargs=kwargs)


# ... to the least specific.
def resolve_str_qalgorithmspec(qalgorithmspec: str) -> QAlgorithm:

    acronym = qalgorithmspec
    try:
        mapping = register[acronym]
    except KeyError:
        raise ValueError

    # TODO: the mapping should also contain information about default keyword
    #       arguments for its `_QModule`s (e.g., default values); we could use
    #       this information for type checking.

    return QAlgorithm(mapping=mapping, kwargs={})


# Finally, we use a strategy pattern to create a unique canonicalisation function ("generic solver").
QAlgorithmSpecSolvers = Enum('QAlgorithmSpecSolvers',
                             [
                                 ('QALGORITHM', resolve_qalgorithm_qalgorithmspec),
                                 ('TUPLE',      resolve_tuple_qalgorithmspec),
                                 ('STR',        resolve_str_qalgorithmspec),
                             ])


def resolve_qalgorithmspec(qalgorithmspec: QAlgorithmSpecType) -> QAlgorithm:

    qalgorithmspec_class = qalgorithmspec.__class__.__name__
    try:
        solver = getattr(QAlgorithmSpecSolvers, qalgorithmspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qalgorithm = solver(qalgorithmspec)
        return qalgorithm
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported QAlgorithm specification type: {qalgorithmspec_class}.")
