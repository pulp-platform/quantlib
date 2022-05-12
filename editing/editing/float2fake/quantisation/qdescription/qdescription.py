from enum import Enum
from typing import NamedTuple, Tuple, Union

from quantlib.algorithms.qbase import QGranularity,         QGranularitySpecType,           resolve_qgranularityspec
from quantlib.algorithms.qbase import QRange,               QRangeSpecType,                 resolve_qrangespec
from quantlib.algorithms.qbase import QHParamsInitStrategy, QHParamsInitStrategySpecType,   resolve_qhparamsinitstrategyspec
from .qalgorithm               import QAlgorithm,           QAlgorithmSpecType,             resolve_qalgorithmspec
from quantlib.utils import quantlib_err_header


# -- CANONICAL DATA STRUCTURE -- #

class QDescription(NamedTuple):
    """Collect pieces of information required to assemble ``_QModule``s.

    ``_QModule``s are meant to be sub-classed. These sub-classes are grouped
    under namespaces identified by the acronym of the PTQ/QAT algorithm that
    they implement.

    Independently of the PTQ/QAT algorithm, each ``_QModule`` must describe
    how it quantises the ``torch.Tensor`` that it is responsible for. This
    description consists of three parts:
    * the quantisation granularity (i.e., which partitions of the
      ``torch.Tensor``'s components use different quantisers);
    * the integer range underlying the fake-quantised representation;
    * the strategy to initialise the quantisers' hyper-parameters.

    Then, each namespace of algorithm-specific ``_QModule``s should define a
    mapping from floating-point ``nn.Module``s to fake-quantised counterparts;
    optionally, each namespace could also define algorithm-specific arguments
    that should be passed to the constructors of its (algorithm-specific)
    ``_QModule`` sub-classes. These two pieces of information are aggregated
    into ``QAlgorithm`` objects.

    """
    qgranularity:         QGranularity
    qrange:               QRange
    qhparamsinitstrategy: QHParamsInitStrategy
    qalgorithm:           QAlgorithm


# -- CANONICALISAITON FLOW -- #

QDescriptionSpecType = Union[QDescription, Tuple[QGranularitySpecType, QRangeSpecType, QHParamsInitStrategySpecType, QAlgorithmSpecType]]


def resolve_qdescription_qdescriptionspectype(qdescriptionspec: QDescription) -> QDescription:
    return qdescriptionspec


def resolve_tuple_qdescriptionspectype(qdescriptionspec: Tuple[QGranularitySpecType, QRangeSpecType, QHParamsInitStrategySpecType, QAlgorithmSpecType]) -> QDescription:

    qgranularityspec, qrangespec, qhparamsinitstrategyspec, qalgorithmspec = qdescriptionspec
    qgranularity = resolve_qgranularityspec(qgranularityspec)
    qrange = resolve_qrangespec(qrangespec)
    qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(qhparamsinitstrategyspec)
    qalgorithm = resolve_qalgorithmspec(qalgorithmspec)

    return QDescription(qgranularity=qgranularity,
                        qrange=qrange,
                        qhparamsinitstrategy=qhparamsinitstrategy,
                        qalgorithm=qalgorithm)


QDescriptionSpecSolvers = Enum('QDescriptionSpecSolvers',
                               [
                                   ('QDESCRIPTION', resolve_qdescription_qdescriptionspectype),
                                   ('TUPLE',        resolve_tuple_qdescriptionspectype),
                               ])


def resolve_qdescriptionspec(qdescriptionspec: QDescriptionSpecType) -> QDescription:

    qdescriptionspec_class = qdescriptionspec.__class__.__name__
    try:
        solver = getattr(QDescriptionSpecSolvers, qdescriptionspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qdescription = solver(qdescriptionspec)
        return qdescription
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported QDescription specification type: {qdescriptionspec_class}.")
