from collections import OrderedDict
import torch.nn as nn
from typing import NamedTuple, Tuple, Dict, Union, Optional, Type, Any

from ...base import Checker


class NNModuleDescription(NamedTuple):
    class_: Type[nn.Module]
    kwargs: Dict[str, Any]
    checkers: Union[Checker, Tuple[Checker, ...]] = tuple()


CandidateName = str
CandidateDescription = Union[NNModuleDescription, None]


class Candidates(OrderedDict):

    def __setitem__(self, name: CandidateName, candidate: CandidateDescription):

        # validate input types
        if not isinstance(name, str):
            raise TypeError
        if not (isinstance(candidate, NNModuleDescription) or (candidate is None)):
            raise TypeError

        super(Candidates, self).__setitem__(name, candidate)
