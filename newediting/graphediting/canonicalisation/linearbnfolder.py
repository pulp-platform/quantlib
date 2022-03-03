import torch.fx as fx
from typing import List, Optional

from ..gebase import GraphRewriter, ApplicationPoint


class LinearBNFolder(GraphRewriter):

    def __init__(self):
        pass

    def find_application_points(self, data_gm: fx.GraphModule) -> List[ApplicationPoint]:
        pass

    def _apply(self, data_gm: fx.GraphModule, ap: ApplicationPoint):
        pass

    def apply(self, data_gm: fx.GraphModule, ap: Optional[ApplicationPoint] = None):
        pass
