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
