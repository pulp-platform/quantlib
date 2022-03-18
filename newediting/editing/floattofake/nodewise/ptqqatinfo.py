from enum import Enum
from typing import Tuple, NamedTuple, Dict, Any, Union

from quantlib.newalgorithms.qalgorithms import ModuleMapping, ptqqat_index
from quantlib.newutils import quantlib_err_header


class PTQQATInfo(NamedTuple):
    mapping: ModuleMapping
    kwargs:  Dict[str, Any]


PTQQATInfoSpecType = Union[PTQQATInfo, Tuple[str, Dict[str, Any]], str]


def resolve_ptqqatinfo_ptqqatinfospec(ptqqatinfospec: PTQQATInfo) -> PTQQATInfo:
    return ptqqatinfospec


def resolve_tuple_ptqqatinfospec(ptqqatinfospec: Tuple[str, Dict[str, Any]]) -> PTQQATInfo:

    algorithm, kwargs = ptqqatinfospec
    try:
        mapping = ptqqat_index.register[algorithm]
    except KeyError:
        raise ValueError()

    return PTQQATInfo(mapping=mapping, kwargs=kwargs)


def resolve_str_ptqqatinfospec(ptqqatinfospec: str) -> PTQQATInfo:

    algorithm = ptqqatinfospec
    try:
        mapping = ptqqat_index.register[algorithm]
    except KeyError:
        raise ValueError()

    kwargs = {}

    return PTQQATInfo(mapping=mapping, kwargs=kwargs)


PTQQATInfoSpecSolvers = Enum('PTQQATInfoSpecSolvers',
                             [
                                 ('PTQQATINFO', resolve_ptqqatinfo_ptqqatinfospec),
                                 ('TUPLE',      resolve_tuple_ptqqatinfospec),
                                 ('STR',        resolve_str_ptqqatinfospec),
                             ])


def resolve_ptqqatinfospec(ptqqatinfospec: PTQQATInfoSpecType) -> PTQQATInfo:

    # I apply a strategy pattern to retrieve the correct solver method based on the PTQQATInfo specification type
    ptqqatinfospec_class = ptqqatinfospec.__class__.__name__
    try:
        solver = getattr(PTQQATInfoSpecSolvers, ptqqatinfospec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        ptqqatinfo = solver(ptqqatinfospec)
        return ptqqatinfo
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported PTQQATInfo specification type: {ptqqatinfospec_class}.")
