from torch import nn
from torch.optim import Adam, SGD, Adagrad

import inspect

from . import pact_ops
from quantlib.editing.lightweight.rules.filters import TypeFilter
from quantlib.editing.lightweight.graph import LightweightGraph

__all__ = ['PACT_Adam', 'PACT_SGD', 'PACT_Adagrad']

_PACT_CLASSES = [cl[1] for cl in inspect.getmembers(pact_ops, inspect.isclass) if issubclass(cl[1], nn.Module)]


# thanks to https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
class PACT_OptimizerFactory:
    def __init__(self):
        self.created_classes = {}
    def __call__(self, base : type):
        rep = "PACT_" + base.__name__
        if rep in self.created_classes.keys():
            return self.created_classes[rep]
        else:
            class PACT_Opt(base):
                def __init__(self, net, pact_decay, *opt_args, **opt_kwargs):
                    pact_filter = TypeFilter(*_PACT_CLASSES)
                    net_nodes = LightweightGraph.build_nodes_list(net)
                    learnable_clip_params = [b for n in pact_filter(net_nodes) for b in n.module.clipping_params.values() if b.requires_grad]
                    # initialize the base class with configured weight decay for the
                    # clipping parameters and any other supplied parameters
                    base.__init__(self,
                                  [{'params':learnable_clip_params,
                                    'weight_decay': pact_decay},
                                   {'params':[p for p in net.parameters() if p.requires_grad and all(pp is not p for pp in learnable_clip_params)]}],
                                  *opt_args,
                                  **opt_kwargs)

            # hide the fact that this class was dynamically created by changing
            # its name and qualname attributes - the perfect crime...
            PACT_Opt.__name__ = rep
            PACT_Opt.__qualname__ = rep
            self.created_classes[rep] = PACT_Opt
            return PACT_Opt

fac = PACT_OptimizerFactory()

PACT_Adam = fac(Adam)
PACT_SGD = fac(SGD)
PACT_Adagrad = fac(Adagrad)
