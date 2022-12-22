import torch
from torch import nn, fx
from quantlib.editing.editing.editors.base.editor import Editor

class BadFlatten1Template(nn.Module):

    # opcode       name    target    args           kwargs
    # -----------  ------  --------  -------------  --------
    # placeholder  x       x         ()             {}
    # call_method  size    size      (x, 0)         {}
    # call_method  view    view      (x, size, -1)  {}
    # output       output  output    (view,)        {}

    def __init__(self):
        super(BadFlatten1Template, self).__init__()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class BadFlatten2Template(nn.Module):

    # opcode       name     target    args        kwargs
    # -----------  -------  --------  ----------  --------
    # placeholder  x        x         ()          {}
    # call_method  flatten  flatten   (x, 1, -1)  {}
    # output       output   output    (flatten,)  {}

    def __init__(self):
        super(BadFlatten2Template, self).__init__()
    
    def forward(self, x):
        x = x.flatten(1, -1)
        return x

class GoodFlattenTemplate(nn.Module):

    # opcode       name     target    args        kwargs
    # -----------  -------  --------  ----------  --------
    # placeholder  x        x         ()          {}
    # call_module  flatten  flatten   (x,)        {}
    # output       output   output    (flatten,)  {}

    def __init__(self):
        super(GoodFlattenTemplate, self).__init__()
        self.flatten = torch.nn.Flatten()
    
    def forward(self, x):
        x = self.flatten(x)
        return x

# TODO: this implementation works, but it uses plain torch.fx instead of
#       the more sophisticated functionality of Rewriters.
#       At this time, I did not manage to get the Rewriters to work with
#       the Flatten patterns, which do not employ module nodes but 
#       method nodes.
class FlattenCanonicaliser(Editor):

    """This editor canonicalises the representation of Flatten nodes.
    """
    def __init__(self):
        super(FlattenCanonicaliser, self).__init__()

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        mbad1 = BadFlatten1Template()
        mbad2 = BadFlatten2Template()
        mgood = GoodFlattenTemplate()
        gbad1 = fx.symbolic_trace(mbad1)
        gbad2 = fx.symbolic_trace(mbad2)
        ggood = fx.symbolic_trace(mgood)
        fx.replace_pattern(g, gbad1, ggood)
        fx.replace_pattern(g, gbad2, ggood)
        return g
        