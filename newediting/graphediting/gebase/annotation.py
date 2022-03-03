import torch.fx as fx


class Annotation:

    def __init__(self):
        pass

    def annotate(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError
