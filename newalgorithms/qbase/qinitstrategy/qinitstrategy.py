from enum import Enum
import inspect
import torch
from typing import Tuple, Dict, Any
from typing import Union, Type
from typing import ClassVar

from ..observer import TensorObserver, MinMaxMeanVarObserver
from quantlib.newutils import quantlib_err_header


class QHParamsInitStrategy(object):

    default_kwargs: ClassVar[Dict[str, Any]] = {}

    def __init__(self):
        super().__init__()

    def get_a_b(self, observer: TensorObserver) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ConstInitStrategy(object):

    default_kwargs = {'a': -1.0, 'b': 1.0}

    def __init__(self, a: float, b: float):
        super().__init__()
        if not a < b:
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"requires a lower and an upper bound, which must be different and passed in this order; instead, I received: ({a}, {b}).")
        self._a = a
        self._b = b

    def get_a_b(self, observer: TensorObserver) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.ones(observer.broadcasting_shape) * self._a
        b = torch.ones(observer.broadcasting_shape) * self._b
        return a, b


class MinMaxInitStrategy(object):

    default_kwargs = {}

    def __init__(self):
        super().__init__()

    def get_a_b(self, observer: MinMaxMeanVarObserver) -> Tuple[torch.Tensor, torch.Tensor]:
        if not observer.is_tracking:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "requires an observer which has collected statistics.")
        a = observer.min
        b = observer.max
        return a, b


class MeanStdInitStrategy(object):

    # We borrow the "golden rule" of Gaussian distributions, where 99.7% of
    # the probability mass lies within 3 standard deviations from the mean.
    default_kwargs = {'n_std': 3}

    def __init__(self, n_std: int):
        super().__init__()
        if n_std < 1:
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"requires a non-negative number of standard deviations, but received the following: {n_std}.")
        self._n_std = n_std

    def get_a_b(self, observer: MinMaxMeanVarObserver) -> Tuple[torch.Tensor, torch.Tensor]:
        if not observer.is_tracking:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "requires an observer which has collected statistics.")
        a = observer.mean - self._n_std * observer.var.sqrt()
        b = observer.mean + self._n_std * observer.var.sqrt()
        return a, b


QHParamsInitStrategySpecType = Union[str, Tuple[str, Dict[str, Any]]]


# define the list of supported strategies, together with their default arguments; then, turn it into an enumerated
QHParamsInitStrategyDefaultParams = Enum('QHParamsInitStrategyDefaultParams',
                                         [
                                             ('CONST',   ConstInitStrategy),
                                             ('MINMAX',  MinMaxInitStrategy),
                                             ('MEANSTD', MeanStdInitStrategy),
                                         ])


def _try_get_qhparamsinitstrategy_class(qhparamsinitstrategyname: str) -> Type[QHParamsInitStrategy]:
    try:
        class_ = getattr(QHParamsInitStrategyDefaultParams, qhparamsinitstrategyname).value
        return class_
    except AttributeError:
        caller_name = inspect.getouterframes(inspect.currentframe())[1].function    # https://stackoverflow.com/a/2529895 and https://docs.python.org/3/library/inspect.html#inspect.getouterframes
        raise ValueError(quantlib_err_header(obj_name=caller_name) + f"does not support the following QHParamsInitStrategy specification: {qhparamsinitstrategyname}.")


QInitStrategySpecType = Union[str, Tuple[str, Dict[str, Any]]]


def resolve_tuple_qhparamsinitstrategyspec(qhparamsinitstrategyspec: Tuple[str, Dict[str, Any]]) -> QHParamsInitStrategy:

    qhparamsinitstrategyname, user_kwargs = qhparamsinitstrategyspec
    qhparamsinitstrategyname = qhparamsinitstrategyname.upper()

    qhparamsinitstrategy_class = _try_get_qhparamsinitstrategy_class(qhparamsinitstrategyname)
    kwargs = qhparamsinitstrategy_class.default_kwargs.copy()
    kwargs.update(user_kwargs)

    supported_keys = set(qhparamsinitstrategy_class.default_kwargs.keys())
    unsupported_keys = set(kwargs.keys()).difference(supported_keys)
    if len(unsupported_keys) > 0:
        raise ValueError(quantlib_err_header() + f"QHParamsInitStrategy {qhparamsinitstrategy_class.__name__} only supports keys {supported_keys}, but I also found these: {unsupported_keys}.")

    qhparamsinitstrategy = qhparamsinitstrategy_class(**kwargs)
    return qhparamsinitstrategy


def resolve_str_qhparamsinitstrategyspec(qhparamsinitstrategyspec: str) -> QHParamsInitStrategy:

    qhparamsinitstrategyname = qhparamsinitstrategyspec.upper()
    mockup_kwargs = {}

    qhparamsinitstrategy = resolve_tuple_qhparamsinitstrategyspec((qhparamsinitstrategyname, mockup_kwargs))

    return qhparamsinitstrategy


QHParamsInitStrategySolvers = Enum('',
                                   [
                                       ('TUPLE', resolve_tuple_qhparamsinitstrategyspec),
                                       ('STR',   resolve_str_qhparamsinitstrategyspec),
                                   ])


def resolve_qhparamsinitstrategyspec(qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QHParamsInitStrategy:

    # I apply a strategy pattern to retrieve the correct solver method based on the QInit specification type
    qhparamsinitstrategyspec_class = qhparamsinitstrategyspec.__class__.__name__
    try:
        solver = getattr(QHParamsInitStrategySolvers, qhparamsinitstrategyspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        qhparamsinitstrategy = solver(qhparamsinitstrategyspec)
        return qhparamsinitstrategy
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported QHParamsInitStrategy specification type: {qhparamsinitstrategyspec_class}.")
