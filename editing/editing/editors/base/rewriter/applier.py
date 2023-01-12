import torch.fx as fx

from .applicationpoint import ApplicationPoint


class Applier(object):

    def __init__(self):
        super(Applier, self).__init__()
        self._counter: int = 0  # use `self._counter` to distinguish applications

    def _apply(self,
               g: fx.GraphModule,
               ap: ApplicationPoint,
               id_: str) -> fx.GraphModule:
        # use `id_` to annotate graph modifications
        raise NotImplementedError

    @staticmethod
    def _polish_fxgraphmodule(g: fx.GraphModule) -> None:
        """Finalise the modifications made in ``_apply``."""
        g.graph.lint()  # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint; this also ensures that `fx.Node`s appear in topological order
        g.recompile()   # https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile

    def apply(self,
              g:   fx.GraphModule,
              ap:  ApplicationPoint,
              id_: str) -> fx.GraphModule:

        # create a unique application identifier
        self._counter += 1
        id_ = id_ + f'_{str(self._counter)}_'

        # modify the graph
        g = self._apply(g, ap, id_)
        Applier._polish_fxgraphmodule(g)

        return g
