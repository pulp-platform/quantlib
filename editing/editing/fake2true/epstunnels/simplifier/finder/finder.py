import torch.fx as fx
from typing import List

from ..applicationpoint import CandidateEpsTunnelConstruct
from .algorithm import find_candidate_constructs, verify_candidate_construct
from quantlib.editing.editing.editors import Finder


class EpsTunnelConstructFinder(Finder):

    def find(self, g: fx.GraphModule) -> List[CandidateEpsTunnelConstruct]:
        candidate_constructs = find_candidate_constructs(g)
        constructs = list(filter(lambda cc: verify_candidate_construct(cc, g), candidate_constructs))
        return constructs

    def check_aps_commutativity(self, aps: List[CandidateEpsTunnelConstruct]) -> bool:
        return True  # TODO: implement the check!
