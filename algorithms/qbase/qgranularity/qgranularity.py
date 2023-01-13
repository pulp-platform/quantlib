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
from typing import Tuple, Union, NewType

from quantlib.utils import quantlib_err_header


# The data type representing quantisation granularity. QuantLib modules will
# instantiate a separate quantiser for each of the specified dimensions. The
# dimensions are specified by means of their integer index. These indices are
# collected in a tuple: this choice implies that the quantisation granularity
# will be static throughout training.
QGranularity = NewType('QGranularity', Tuple[int, ...])


# The data type representing how users can specify quantisation granularity:
#   * either they provide the explicit collection of dimension indices
#   * or they provide string shortcuts.
QGranularitySpecType = Union[QGranularity, Tuple[int, ...], str]


def resolve_qgranularity_qgranularityspec(qgranularityspec: QGranularity) -> QGranularity:
    return qgranularityspec


def resolve_tuple_qgranularityspec(qgranularityspec: Tuple[int, ...]) -> QGranularity:
    """Map an explicit index collection (tuple of integers) to a
    ``QGranularity`` object.
    """
    if any(map(lambda i: i < 0, qgranularityspec)):
        raise ValueError(quantlib_err_header() + f"requires non-negative dimension indices, but found the following values: {tuple(i for i in qgranularityspec if i < 0)}.")

    qgranularityspec = tuple(set(qgranularityspec))  # remove duplicate values
    qgranularityspec = tuple(sorted(qgranularityspec))
    qgranularity = QGranularity(qgranularityspec)
    return qgranularity


# String shortcuts must belong to the following list. We assume the NCHW
# ordering of PyTorch.
QGranularityStrSpecOptions = Enum('QGranularityStrSpecOptions',
                                  [
                                      ('PER-ARRAY',              QGranularity(tuple())),
                                      ('PER-OUTCHANNEL_WEIGHTS', QGranularity((0,))),
                                  ])


def resolve_str_qgranularityspec(qgranularityspec: str) -> QGranularity:
    """Map a (supported) string shortcut to a ``QGranularity`` object."""
    qgranularityspec = qgranularityspec.upper()
    try:
        qgranularity = getattr(QGranularityStrSpecOptions, qgranularityspec).value
        return qgranularity
    except AttributeError:
        raise ValueError(f"unsupported QGranularity string specification: {qgranularityspec}.")


# Redirect the general solver to the type-specific solver method.
QGranularitySpecSolvers = Enum('QGranularitySpecSolvers',
                               [
                                   ('QGRANULARITY', resolve_qgranularity_qgranularityspec),
                                   ('TUPLE',        resolve_tuple_qgranularityspec),
                                   ('STR',          resolve_str_qgranularityspec),
                               ])


def resolve_qgranularityspec(qgranularityspec: QGranularitySpecType) -> QGranularity:
    """Resolve user specifications into ``QGranularity`` objects.

    This function uses a strategy pattern to disambiguate different types of
    specifications.
    """
    qgranularityspec_class = qgranularityspec.__class__.__name__.upper()

    try:
        solver = getattr(QGranularitySpecSolvers, qgranularityspec_class)
        qgranularity = solver(qgranularityspec)
        return qgranularity
    except AttributeError:
        raise TypeError(f"unsupported QGranularity specification type: {qgranularityspec_class}.")
