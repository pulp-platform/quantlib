from enum import Enum
from typing import NamedTuple
from typing import Tuple, Dict, Any
from typing import Union

from quantlib.newutils import quantlib_err_header


class QInitStrategy(NamedTuple):
    """A strategy name and its (potentially empty) set of named arguments."""
    name: str
    kwargs: Dict[str, Any]


# define the list of supported strategies, together with their default arguments; then, turn it into an enumerated
QInitStrategies = Enum('QInitStrategies',
                       [
                           ('CONST',   {'lo': -1.0, 'hi': 1.0}),
                           ('MINMAX',  {}),
                           ('MEANSTD', {'n_std': 3}),
                       ])


def _try_get_default_kwargs(name: str) -> Dict[str, Any]:
    try:
        default_kwargs = QInitStrategies[name].value
    except KeyError:
        raise ValueError(quantlib_err_header() + f"Unsupported QInitStrategy: {name}.")
    return default_kwargs


QInitStrategySpecType = Union[str, Tuple[str, Dict[str, Any]]]


def resolve_str_qinitspec(qinitspec: str) -> QInitStrategy:

    name = qinitspec.upper()

    default_kwargs = _try_get_default_kwargs(name)
    kwargs = default_kwargs.copy()

    return QInitStrategy(name=name, kwargs=kwargs)


def resolve_tuple_qinitspec(qinitspec: Tuple[str, Dict[str, Any]]) -> QInitStrategy:

    name, user_kwargs = qinitspec
    name = name.upper()

    default_kwargs = _try_get_default_kwargs(name)
    kwargs = default_kwargs.copy()
    kwargs.update(user_kwargs)

    supported_keys = set(default_kwargs.keys())
    unsupported_keys = set(kwargs.keys()).difference(supported_keys)
    if len(unsupported_keys) > 0:
        raise ValueError(quantlib_err_header() + f"QInitStrategy {name} only supports keys {supported_keys}, but I also found these: {unsupported_keys}.")

    return QInitStrategy(name=name, kwargs=kwargs)


class QInitStrategySpec(Enum):
    TUPLE = resolve_tuple_qinitspec
    STR   = resolve_str_qinitspec


def resolve_qinitspec(qinitspec: QInitStrategySpecType) -> QInitStrategy:

    # I apply a strategy pattern to retrieve the correct solver method based on the QInit specification type
    qinitspec_class = qinitspec.__class__.__name__
    try:
        solver = getattr(QInitStrategySpec, qinitspec_class.upper())  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
    except KeyError:
        raise TypeError(quantlib_err_header() + f"Unsupported QInitStrategy specification type: {qinitspec_class}.")

    qinit = solver(qinitspec)
    return qinit



# import enum
#
# from ..qhparams import QGranularity
#
#
# @enum.unique
# class QUnquantisedBounds(enum.Enum):
#     CONST = 0
#     MINMAX = 1
#     DIST = 3
#
# if strategy is QUnquantisedBounds.CONST:
#     clip_lo = torch.ones_like(self._min) * -1.0
#     clip_hi = torch.ones_like(self._max) * 1.0
#
# elif strategy is QUnquantisedBounds.MINMAX:
#     clip_lo = self._min
#     clip_hi = self._max
#
# elif strategy is QUnquantisedBounds.DIST:
#     # I borrow the "golden rule" of Gaussian distributions, where
#     # 99.7% of the probability mass lies within three standard
#     # deviations from the mean.
#     n_std = 3
#     clip_lo = self._mean - n_std * self._var.sqrt()
#     clip_hi = self._mean + n_std * self._var.sqrt()
#
# else:
#     raise ValueError(f"[QuantLab] QStatisticsTracker can not return distribution bounds with strategy {strategy}.")
