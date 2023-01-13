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
from typing import Tuple, Dict
from typing import Union

from quantlib.utils import UnknownType, UNKNOWN
from quantlib.utils import quantlib_err_header


# aliases (for readability)
IMPLICIT_STEP = 1


class QRange(object):
    """A class to represent integer indexings of bin collections.

    Quantisation if a form of information compression. Given a base set
    :math:`X`, we can quantise it into :math:`K > 1` bins by creating or
    finding a partition :math:`\{ X_{0}, \dots, X_{K-1} \}`. From this
    perspective, a quantisation is a mapping

    .. math::
        f \,:\, X &\to     \{ \lambda_{0}, \dots, \lambda_{K-1} \} \\
                x &\mapsto a_{k} \iff \chi_{X_{k}}(x) = 1 \,,

    where :math:`\{ \lambda_{0}, \dots, \lambda_{K-1} \}` is an abstract
    collection of bin labels and :math:`\chi_{X_{k}}` is the indicator
    function of the :math:`k`-th subset of :math:`X`.

    In the specific case of real numbers, we are interested in quantisations
    which preserve the semantics of the input; i.e., bin labels should convey
    information about the numbers that are contained in the bin. Therefore,
    common quantisations of the real numbers consist of consecutive intervals
    which cover the whole real line, and each interval is mapped to some
    statistic associated with its contained numbers; for instance, each number
    is mapped to the mean of its encolsing interval.

    Any quantisation operation over real numbers can be decomposed into three
    operations:

    * binning: mapping the input real number to the bin label associated with
      the element of the partition to which it belongs; the bin label is a
      categorical value taken from a finite set, and note that it is not
      necessary that this value is a number;
    * re-indexing or re-labelling: mapping the categorical bin label to an
      integer index;
    * de-quantisation: mapping the integer index to a real value.

    Most quantisation algorithms targetting neural networks often fuse one or
    more of these steps. The most common fusion is that of binning and
    re-indexing, since it can be realised using efficient HW operations such
    as rounding and flooring.

    This class provides a way to describe re-indexing ranges in such a way to
    convey the semantics of the output of the de-quantisation. For instance:

    * indexing ranges starting from zero suggest that the re-indexing range is
      a subset of some unsigned integer digital data type;
    * quasi-symmetric indexing ranges (e.g., :math:`(-4, 2, \dots, 3)`)
      suggest that the re-indexing range is a subset of some signed integer
      digital data type using two's complement representation;
    * symmetric indexing ranges (e.g., :math:`(-3, 2, \dots, 3)`) suggest that
      that the re-indexing range is a subset of some signed integer digital
      data type using sign-magnitude representation as opposed to two's
      complement.

    We also allow for the special "sign" indexing range :math:`(-1, 1)`.

    """
    def __init__(self, offset: Union[int, UnknownType], n_levels: int, step: int):
        """Create a QRange object.

        ``QRange``s are the abstraction by which QuantLib users describe
        quantisers to ``QModule``s. Note that this information should not
        change after being specified. We make this choice since we assume that
        users know how they would like to quantise their neural network before
        running a single training iteration.

        """

        self._offset = offset
        self._n_levels = n_levels
        self._step = step

    @property
    def offset(self) -> Union[int, UnknownType]:
        return self._offset

    @property
    def n_levels(self) -> int:
        return self._n_levels

    @property
    def step(self) -> int:
        return self._step

    @property
    def range(self) -> Union[Tuple[int, ...], UnknownType]:
        try:
            return tuple(range(self._offset, self._offset + self._n_levels * self._step, self._step))
        except TypeError:  # `self._offset == UNKNOWN`
            return UNKNOWN

    @property
    def min(self) -> Union[int, UnknownType]:
        try:
            return self.range[0]
        except TypeError:
            return UNKNOWN

    @property
    def max(self) -> Union[int, UnknownType]:
        try:
            return self.range[-1]
        except TypeError:
            return UNKNOWN

    @property
    def is_sign_range(self) -> bool:
        """Signal the intention of the user to specify the binary sign range."""
        try:
            return self.range == (-1, 1)
        except TypeError:
            return False  # the sign range always specifies the offset, so it does not raises an exception

    @property
    def is_unsigned(self) -> bool:
        """Signal the intention of the user to specify a UINT sub-range."""
        try:
            return (self.min == 0) and (self.step == 1)
        except TypeError:
            return False  # unsigned ranges always specify their offset, so they do not raise an exception

    @property
    def is_quasisymmetric(self) -> bool:
        """Signal the intention of the user to specify a two's complement INT
        sub-range."""
        try:
            return (abs(self.min) == abs(self.max) + 1) and (self.step == 1)
        except TypeError:
            return False  # signed ranges always specify their offset, so they do not raise an exception

    @property
    def is_symmetric(self):
        """Signal the intention of the user to specify a sign-magnitude INT
        sub-range."""
        try:
            return self.is_sign_range or ((abs(self.min) == abs(self.max)) and (self.step == 1))
        except TypeError:
            return False  # signed ranges always specify their offset, so they do not raise an exception


QRangeSpecType = Union[QRange, Tuple[int, ...], Dict[str, int], str]


def resolve_qrange_qrangespec(qrangespec: QRange) -> QRange:
    return qrangespec


def resolve_tuple_qrangespec(qrangespec: Tuple[int, ...]) -> QRange:

    # I allow for non-sorted tuples
    qrangespec = tuple(sorted(qrangespec))

    # "degenerate" binary range
    if qrangespec == (-1, 1):
        step = 2
        n_levels = len(qrangespec)
        offset = -1

    else:

        # check spacing step
        steps = set([j - i for i, j in zip(qrangespec[:-1], qrangespec[1:])])
        if len(steps) > 1:
            raise ValueError(quantlib_err_header() + f"QRange tuple specifications should be composed of equally-spaced integers, but multiple steps were specified: {steps}.")
        else:
            step = steps.pop()
            if step != IMPLICIT_STEP:
                raise ValueError(quantlib_err_header() + f"QRange tuple specifications which are not (-1, 1) must have a step of {IMPLICIT_STEP}, but {step} was specified.")

        n_levels = len(qrangespec)
        offset = min(qrangespec)

    return QRange(offset, n_levels, step)


def resolve_dict_qrangespec(qrangespec: Dict[str, int]) -> QRange:

    step = IMPLICIT_STEP

    n_levels_keys = {'n_levels', 'bitwidth', 'limpbitwidth'}
    offset_keys = {'offset', 'signed'}

    # check that the keys conform to the specification
    qrangespec_keys = set(qrangespec.keys())
    unknown_keys = qrangespec_keys.difference(n_levels_keys | offset_keys)
    if len(unknown_keys) != 0:
        raise ValueError(quantlib_err_header() + f"QRange dictionary specification does not support the following keys: {unknown_keys}.")

    # canonicalise number of levels
    qrangespec_n_levels_keys = qrangespec_keys.intersection(n_levels_keys)

    if len(qrangespec_n_levels_keys) == 0:
        raise ValueError(quantlib_err_header() + f"QRange dictionary specification must specify at least one of the following keys: {n_levels_keys}.")

    elif len(qrangespec_n_levels_keys) == 1:
        if qrangespec_n_levels_keys == {'bitwidth'}:
            n_levels = 2 ** qrangespec['bitwidth']

        elif qrangespec_n_levels_keys == {'limpbitwidth'}:
            n_levels = 2 ** qrangespec['limpbitwidth'] - 1

        else:  # qrangespec_n_levels_keys == {'n_levels'}
            n_levels = qrangespec['n_levels']

    else:
        raise ValueError(quantlib_err_header() + f"QRange dictionary specification specified the number of levels ambiguously: {qrangespec_n_levels_keys}.")

    # canonicalise offset
    qrangespec_offset_keys = qrangespec_keys.intersection(offset_keys)

    if len(qrangespec_offset_keys) == 0:
        offset = UNKNOWN

    elif len(qrangespec_offset_keys) == 1:

        if qrangespec_offset_keys == {'signed'}:
            offset = -(n_levels // 2) if qrangespec['signed'] is True else 0

        else:  # offset_qrange_keys == {'offset'}
            offset = qrangespec['offset']

    else:
        raise ValueError(quantlib_err_header() + f"QRange dictionary specification specified the offset ambiguously: {qrangespec}.")

    return QRange(offset, n_levels, step)


# String shortcuts must belong to the following list. We assume the NCHW
# ordering of PyTorch.
QRangeStrSpecOptions = Enum('QRangeStrSpecOptions',
                            [
                                ('BINARY',  QRange(offset=-1, n_levels=2, step=2)),
                                ('TERNARY', QRange(offset=-1, n_levels=3, step=IMPLICIT_STEP)),
                            ])


def resolve_str_qrangespec(qrangespec: str) -> QRange:
    """Map a (supported) string shortcut to a ``QGranularity`` object."""
    qrangespec = qrangespec.upper()
    try:
        qrange = getattr(QRangeStrSpecOptions, qrangespec).value
        return qrange
    except AttributeError:
        raise ValueError(f"unsupported QGranularity string specification: {qrangespec}.")


QRangeSpecSolvers = Enum('QRangeSpecSolvers',
                         [
                             ('QRANGE', resolve_qrange_qrangespec),
                             ('TUPLE',  resolve_tuple_qrangespec),
                             ('DICT',   resolve_dict_qrangespec),
                             ('STR',    resolve_str_qrangespec),
                         ])


def resolve_qrangespec(qrangespec: QRangeSpecType) -> QRange:
    """A function to canonicalise specifications of integer ranges.

    During my experience in research on quantised neural networks I have come
    across several ways of describing integer ranges.

    * The ``UINTB`` integer ranges are ranges of consecutive positive integers
      starting from zero. They can be specified via the positive integer
      bitwidth :math:`B`.
    * The ``INTB`` integer ranges are ranges of consecutive integers starting
      from some negative number. These ranges can be described using the two's
      complement representation, and can be specified via the positive integer
      bitwidth :math:`B`.
    * The "limp" ``INTB`` integer ranges are ranges of consecutive integers
      starting from some negative number. These ranges can be described in
      digital arithmetic using the sign-magnitude representation (therefore
      they can represent one less value than their two's complement
      counterparts), and can be specified via a positive integer bitwidth
      :math:`B`. In particular, the ternary range :math:`\{ -1, 0, 1 \}`
      can be represented as "limp" ``INT2``.
    * Explicit enumerations of :math:`K` consecutive integers starting from an
      integer offset :math:`z`.
    * The special sign range :math:`\{ -1, 1 \}`, which differs from both the
      ``UINT1`` range (:math:`\{ 0, 1 \}`) and the ``INT1`` range (:math:`\{
      -1, 0 \}`).

    In particular, the integer ranges specified by ``UINTB``, ``INTB`` and
    "limp" ``INTB`` are just some of all the possible finite ranges of
    consecutive integers. Therefore, specifying an integer range using an
    explicit number of levels and an explicit offset is more general.

    However, note also that spaced-by-one integer ranges can always be
    immersed into some digital integer range; for instance, the range
    :math:`\{ -1, 0, 1, 2 \}` is a subset of the ``INT4`` range :math:`\{ -8,
    -7, \dots, 7 \}`.

    In QuantLib, we allow for several ways of specifying integer ranges.

    * Explicit tuple-based specifications. We allow to express integer ranges
      in extensive form, representing them as tuples of integers. Such tuples
      can be passed not sorted, but once sorted their components must have a
      step of one. The only exception to the step constraint is the special
      sign range :math:`\{ -1, 1 \}`.
    * Compact dictionary-based specifications. We allow for several formats,
      all of which should be resolved to a positive number of levels :math:`K`
      and an integer offset :math:`z`. There are three mutually exclusive ways
      to specify :math:`K`:
      * an explicit number of levels :math:`K` (``n_levels``, positive
        integer);
      * a bitwidth :math:`B` (``bitwidth``, positive integer), so that
        :math:`K = 2^{B}`;
      * a "limp" bitwidth :math:`B` (``limpbitwidth``, positive integer);
        :math:`K = 2^{B} - 1`.
      There are two mutually exclusive ways to specify :math:`z`:
      * an explicit integer offset :math:`z` (``offset``, integer);
      * signedness information (``signed``, Boolean); if unsigned, :math:`z`
        is set to zero; if signed and :math:`K` is even, :math:`z` is set to
        :math:`-K / 2`; if signed and :math:`K` is odd, :math:`z` is set to
        :math:`-(K - 1) / 2.
    * Compact string-based specifications. We allow for some syntactic sugar
      to express a limited number of integer ranges:
      * ``binary`` creates the range :math:`\{ -1, 1 \}`;
      * ``ternary`` creates the range :math:`\{ -1, 0, 1 \}`.

    """

    # I apply a strategy pattern to retrieve the correct solver method based on the QRange specification type
    qrangespec_class = qrangespec.__class__.__name__.upper()
    try:
        solver = getattr(QRangeSpecSolvers, qrangespec_class)  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qrange = solver(qrangespec)
        return qrange
    except AttributeError:
        raise TypeError(quantlib_err_header() + f"Unsupported QRange specification type: {qrangespec_class}.")
