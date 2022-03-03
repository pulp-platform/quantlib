import torch.fx as fx
from typing import List, Any


class Rewriting:

    def __init__(self):
        pass

    def find(self, g: fx.GraphModule) -> List[Any]:
        raise NotImplementedError

    def rewrite(self, g: fx.GraphModule, ap: Any) -> fx.GraphModule:
        raise NotImplementedError
