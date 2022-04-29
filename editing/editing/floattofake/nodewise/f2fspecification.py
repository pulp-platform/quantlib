from enum import Enum
from typing import Tuple, List, NamedTuple, Union

from .modulefilter                import ModuleFilterSpecType,         resolve_modulefilterspec,         ModuleFilter
from .ptqqatinfo                  import PTQQATInfoSpecType,           resolve_ptqqatinfospec,           PTQQATInfo
from quantlib.algorithms.qbase import QRangeSpecType,               resolve_qrangespec,               QRange
from quantlib.algorithms.qbase import QGranularitySpecType,         resolve_qgranularityspec,         QGranularity
from quantlib.algorithms.qbase import QHParamsInitStrategySpecType, resolve_qhparamsinitstrategyspec, QHParamsInitStrategy

from quantlib.utils import quantlib_err_header


class F2FPartition(NamedTuple):
    modulefilter:         ModuleFilter
    qrange:               QRange
    qgranularity:         QGranularity
    qhparamsinitstrategy: QHParamsInitStrategy
    ptqqatinfo:           PTQQATInfo


F2FPartitionSpecType = Union[F2FPartition, Tuple[ModuleFilterSpecType, QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType, PTQQATInfoSpecType]]


def resolve_f2fpartition_f2fpartitionspec(f2fpartitionspec: F2FPartition) -> F2FPartition:
    return f2fpartitionspec


def resolve_tuple_f2fpartitionspec(f2fpartitionspec: F2FPartitionSpecType) -> F2FPartition:

    modulefilter         = resolve_modulefilterspec(f2fpartitionspec[0])
    qrange               = resolve_qrangespec(f2fpartitionspec[1])
    qgranularity         = resolve_qgranularityspec(f2fpartitionspec[2])
    qhparamsinitstrategy = resolve_qhparamsinitstrategyspec(f2fpartitionspec[3])
    ptqqatinfo           = resolve_ptqqatinfospec(f2fpartitionspec[4])

    f2fpartition = F2FPartition(modulefilter=modulefilter,
                                qrange=qrange,
                                qgranularity=qgranularity,
                                qhparamsinitstrategy=qhparamsinitstrategy,
                                ptqqatinfo=ptqqatinfo)

    return f2fpartition


F2FPartitionSpecSolvers = Enum('F2FPartitionSpecSolvers',
                               [
                                   ('F2FPARTITION', resolve_f2fpartition_f2fpartitionspec),
                                   ('TUPLE',        resolve_tuple_f2fpartitionspec),
                               ])


def resolve_f2fpartitionspec(f2fpartitionspec: F2FPartitionSpecType) -> F2FPartition:

    # I apply a strategy pattern to retrieve the correct solver method based on the F2FPartition specification type
    f2fpartitionspec_class = f2fpartitionspec.__class__.__name__
    try:
        solver = getattr(F2FPartitionSpecSolvers, f2fpartitionspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        f2fpartition = solver(f2fpartitionspec)
        return f2fpartition
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported F2FPartition specification type: {f2fpartitionspec_class}.")


F2FSpecType = List[F2FPartitionSpecType]


def resolve_f2fspec(f2fspec: F2FSpecType) -> List[F2FPartition]:
    return list(map(lambda f2fpartitionspec: resolve_f2fpartitionspec(f2fpartitionspec), f2fspec))
