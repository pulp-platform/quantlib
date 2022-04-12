from torch.optim import SGD, Adam, Adagrad

from .bb_ops import _BB_CLASSES

from quantlib.editing.lightweight.rules.filters import TypeFilter, VariadicOrFilter
from quantlib.editing.lightweight.graph import LightweightGraph
from quantlib.algorithms.pact import PACTUnsignedAct

__all__ = ["BBSGD",
           "BBAdam",
           "BBAdagrad"]

class BBOptimizerFactory(object):
    def __init__(self):
        self._created_classes = {}

    def __call__(self, base_opt_type : type):
        rep = "BB" + base_opt_type.__name__
        if rep in self._created_classes.keys():
            return self._created_classes[rep]
        else:
            class BBOptimizer(base_opt_type):

                def __init__(self, net, gate_lr, pact_decay=0.0, **opt_kwargs):
                    gate_parameters = list(set([p for n, p in net.named_parameters() if n.endswith("bb_gates")]))
                    net_nodes   = LightweightGraph.build_nodes_list(net)
                    pact_filter = TypeFilter(PACTUnsignedAct)
                    learnable_clipping_params = [b for n in pact_filter(net_nodes) for k, b in n.module.clipping_params.items() if b.requires_grad and k != 'log_t']
                    other_params = [p for p in net.parameters() if all(p is not pp for pp in gate_parameters)]
                    base_opt_type.__init__(self,
                                           ({'params': other_params},
                                            {'params': learnable_clipping_params,
                                             'weight_decay': pact_decay}),
                                           **opt_kwargs)
                    self.adam = Adam(params=gate_parameters, lr=gate_lr)

                def state_dict(self):
                    state = {}
                    state['base_optimizer'] =  base_opt_type.state_dict(self)
                    state['adam'] = self.adam.state_dict()
                    return state

                def load_state_dict(self, state_dict):
                    base_opt_type.load_state_dict(self, state_dict['base_optimizer'])
                    self.adam.load_state_dict(state_dict['adam'])

                def step(self, closure=None):
                    base_opt_type.step(self, closure=closure)
                    self.adam.step(closure=closure)

            BBOptimizer.__name__ = rep
            BBOptimizer.__qualname__ = rep
            self._created_classes[rep] = BBOptimizer
            return BBOptimizer


opt_factory = BBOptimizerFactory()

BBSGD = opt_factory(SGD)
BBAdam = opt_factory(Adam)
BBAdagrad = opt_factory(Adagrad)
