import torch.fx as fx
from typing import List

from .applicationpoint import ApplicationPoint


class Finder(object):

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        raise NotImplementedError

    def check_aps_commutativity(self, aps: List[ApplicationPoint]) -> bool:
        """Verify that the application points do not overlap.

        Passing this test ensures that the rewritings of the different
        application points can commute. Therefore, this avoids the need of
        recomputing the application points in-between applications of the same
        ``Rewriter``.

        """
        raise NotImplementedError
