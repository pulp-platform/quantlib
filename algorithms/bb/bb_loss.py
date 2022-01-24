from torch import nn
from quantlib.editing.lightweight.rules.filters import TypeFilter, VariadicOrFilter
from quantlib.editing.lightweight.graph import LightweightGraph
from .bb_ops import _BB_CLASSES

__all__ = ["BBMSELoss",
           "BBCrossEntropyLoss"]

# yeah dynamic inheritance!

class BBLossFactory(object):
    def __init__(self):
        self._created_classes = {}

    def __call__(self, base_loss_type : type):
        rep = "BB" + base_loss_type.__name__
        if rep in self._created_classes.keys():
            return self._created_classes[rep]
        else:
            class BBLoss(base_loss_type):
                def __init__(self, net, mu0, *args, **kwargs):
                    net_nodes   = LightweightGraph.build_nodes_list(net)
                    bb_filter = VariadicOrFilter(*[TypeFilter(t) for t in _BB_CLASSES])
                    bb_modules = [n.module for n in bb_filter(net_nodes)]
                    self.controllers = list(set([m.gate_ctrl for m in bb_modules if m.gate_ctrl is not None]))
                    self.mu0 = mu0
                    base_loss_type.__init__(self, *args, **kwargs)

                def forward(self, *args, **kwargs):
                    loss = base_loss_type.forward(self, *args, **kwargs)
                    bb_loss = 0.
                    for c in self.controllers:
                        bb_loss = bb_loss + self.mu0 * c.loss_term()
                    print(f"bb_loss: {bb_loss}")
                    return loss + bb_loss

            BBLoss.__name__ = rep
            BBLoss.__qualname__ = rep
            self._created_classes[rep] = BBLoss
            return BBLoss

loss_factory = BBLossFactory()
BBMSELoss = loss_factory(nn.MSELoss)
BBCrossEntropyLoss = loss_factory(nn.CrossEntropyLoss)
