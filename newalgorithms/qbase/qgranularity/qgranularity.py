from enum import Enum
from typing import Tuple, Union, NewType

from quantlib.newutils import quantlib_err_header


# The data type representing quantisation granularity. QuantLib modules will
# instantiate a separate quantiser for each of the specified dimensions. The
# dimensions are specified by means of their integer index. These indices are
# collected in a tuple: this choice implies that the quantisation granularity
# will be static throughout training.
QGranularity = NewType('QGranularity', Tuple[int, ...])


# The data type representing how users can specify quantisation granularity:
#   * either they provide the explicit collection of dimension indices
#   * or they provide string shortcuts.
QGranularitySpecType = Union[Tuple[int, ...], str]


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
                                   ('TUPLE', resolve_tuple_qgranularityspec),
                                   ('STR',   resolve_str_qgranularityspec),
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
