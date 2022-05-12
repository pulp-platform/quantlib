import torch.fx as fx


class Editor(object):

    def __init__(self):
        super(Editor, self).__init__()

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        raise NotImplementedError

    def __call__(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        return self.apply(g, *args, **kwargs)
