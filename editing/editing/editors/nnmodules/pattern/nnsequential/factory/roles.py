from collections import OrderedDict
import itertools
from typing import Tuple

from .candidates import CandidateName, Candidates


Role       = str
Screenplay = Tuple[CandidateName, ...]


class Roles(OrderedDict):

    @property
    def roles(self) -> Tuple[Role, ...]:
        return tuple(self.keys())

    @property
    def all_screenplays(self) -> Tuple[Screenplay, ...]:
        role_candidates = []
        for name_to_candidate in self.values():
            role_candidates.append(tuple(name_to_candidate.keys()))
        return tuple(itertools.product(*role_candidates))

    def check_screenplay(self, input_: Screenplay) -> None:

        if not (isinstance(input_, tuple) and all(isinstance(item_, str) for item_ in input_)):
            raise TypeError
        if not len(input_) == len(self.roles):
            raise ValueError  # some roles can't be covered
        if not all(name in self[role].keys() for role, name in zip(self.roles, input_)):
            raise ValueError  # candidate not found

    def __setitem__(self, role: Role, candidates: Candidates):

        # validate input types
        if not isinstance(role, str):
            raise TypeError
        if not isinstance(candidates, Candidates):
            raise TypeError

        super(Roles, self).__setitem__(role, candidates)
