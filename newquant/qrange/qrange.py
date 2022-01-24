import enum
from typing import Tuple, Dict
from typing import Union

from quantlib.newutils import quantlib_err_msg


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
    def __init__(self, n_levels: int, offset: int, step: int):

        self._n_levels = n_levels
        self._offset = offset
        self._step = step

    @property
    def range(self) -> Tuple[int, ...]:
        return tuple(range(self._offset, self._offset + self._n_levels * self._step, self._step))

    @property
    def n_levels(self) -> int:
        return self._n_levels

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def step(self) -> int:
        return self._step

    @property
    def is_sign_range(self) -> int:
        return self.range == (-1, 1)

    @property
    def min(self) -> int:
        return self.range[0]

    @property
    def max(self) -> int:
        return self.range[-1]

    @property
    def is_unsigned(self):
        return (not self.is_sign_range) and (self.min == 0)

    @property
    def is_quasisymmetric(self):
        return (not self.is_sign_range) and (abs(self.max) - abs(self.min) == -1)

    @property
    def is_symmetric(self):
        return (not self.is_sign_range) and (abs(self.min) == abs(self.max))


# I define a QRangeSpec as the union data type `QuantSpec := Union[Tuple[int, ...], Dict[str, int], str]`.

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
            raise ValueError(quantlib_err_msg() + f"QRange tuple specifications should be composed of equally-spaced integers, but multiple steps were specified ({steps}).")
        else:
            step = steps.pop()
            if step != 1:
                raise ValueError(quantlib_err_msg() + f"QRange tuple specifications which are not (-1, 1) must have a step of one (step {step} was specified).")

        n_levels = len(qrangespec)
        offset = min(qrangespec)

    return QRange(n_levels, offset, step)


def resolve_dict_qrangespec(qrangespec: Dict[str, int]):

    step = 1

    n_levels_keys = {'n_levels', 'bitwidth', 'limpbitwidth'}
    offset_keys = {'offset', 'signed'}

    # check that the keys conform to the specification
    qrangespec_keys = set(qrangespec.keys())
    unknown_keys = qrangespec_keys.difference(n_levels_keys | offset_keys)
    if len(unknown_keys) != 0:
        raise ValueError(quantlib_err_msg() + f"QRange dictionariy specification does not support the following keys: {unknown_keys}.")

    # canonicalise number of levels
    qrangespec_n_levels_keys = qrangespec_keys.intersection(n_levels_keys)

    if len(qrangespec_n_levels_keys) == 0:
        raise ValueError(quantlib_err_msg() + f"QRange dictionary specification must specify at least one of the following keys: {n_levels_keys}.")

    elif len(qrangespec_n_levels_keys) == 1:
        if qrangespec_n_levels_keys == {'bitwidth'}:
            n_levels = 2 ** qrangespec['bitwidth']

        elif qrangespec_n_levels_keys == {'limpbitwidth'}:
            n_levels = 2 ** qrangespec['limpbitwidth'] - 1

        else:  # qrangespec_n_levels_keys == {'n_levels'}
            n_levels = qrangespec['n_levels']

    else:
        raise ValueError(quantlib_err_msg() + f"QRange dictionary specification specified the number of levels ambiguously: {qrangespec_n_levels_keys}.")

    # canonicalise offset
    qrangespec_offset_keys = qrangespec_keys.intersection(offset_keys)

    if len(qrangespec_offset_keys) == 0:
        raise ValueError(quantlib_err_msg() + f"QRange dictionary specification must specify at least one of the following keys: {offset_keys}.")

    elif len(qrangespec_offset_keys) == 1:

        if qrangespec_offset_keys == {'signed'}:
            offset = -(n_levels // 2) if qrangespec['signed'] is True else 0

        else:  # offset_qrange_keys == {'offset'}
            offset = qrangespec['offset']

    else:
        raise ValueError(quantlib_err_msg() + f"QRange dictionary specification specified the offset ambiguously: {qrangespec}.")

    return QRange(n_levels, offset, step)


def resolve_str_qrangespec(qrangespec: str) -> QRange:

    if qrangespec == 'binary':
        step = 2
        offset = -1
        n_levels = 2

    elif qrangespec == 'ternary':
        step = 1
        offset = -1
        n_levels = 3

    else:
        raise ValueError(quantlib_err_msg() + f"QRange string specification does not support the following key: {qrangespec}.")

    return QRange(n_levels, offset, step)


@enum.unique
class QRangeSpec(enum.Enum):
    TUPLE = resolve_tuple_qrangespec
    DICT = resolve_dict_qrangespec
    STR = resolve_str_qrangespec


def resolve_qrangespec(qrangespec: Union[Tuple[int, ...], Dict[str, int], str]) -> QRange:
    """A function to disambiguate user specifications of integer ranges.

    During my experience in research on quantised neural networks I have come
    across of several ways of describing integer ranges.

    * The ``UINTX`` integer range represents positive integers, and can be
      specified via the positive integer bitwidth ``X``.
    * The ``INTX`` integer range represents integers using the two's
      complement representation and can be specified via the positive integer
      bitwidth ``X``.
    * The "limp" ``INTX`` integer range represents integers using the
      sign-magnitude representation, and can therefore represent one less
      value than its two's complement counterpart (zero is signed in this
      representation). In particular, the ternary range :math:`\{ -1, 0, 1 \}`
      can be represented as "limp" ``INT2``.
    * Explicit enumerations of consecutive integers starting from an integer
      offset :math:`z`.
    * The special sign range :math:`\{ -1, 1 \}`, which differs from both the
      UINT1 range (:math:`\{ 0, 1 \}`) and the INT1 range (:math:`\{ -1, 0
      \}`).

    In particular, the integer ranges specified by ``UINTX``, ``INTX`` and
    "limp" ``INTX`` are just some of all the possible finite integer ranges
    whose items are spaced by one. Therefore, specifying an integer range
    using an explicit number of levels and an offset is more general.

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
      all of which should express a positive number of levels :math:`K` and an
      integer offset :math:`z`. There are three mutually exclusive ways to
      specify :math:`K`:
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
    qrangespec_class = qrangespec.__class__.__name__
    try:
        solver = getattr(QRangeSpec, qrangespec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
    except KeyError:
        raise KeyError(quantlib_err_msg() + f"Unsupported QRange specification type: {qrangespec_class}.")

    qrange = solver(qrangespec)
    return qrange
