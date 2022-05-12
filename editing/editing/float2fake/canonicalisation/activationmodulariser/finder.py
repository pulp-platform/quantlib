import torch.fx as fx
from typing import List

from .applicationpoint import ActivationNode
from .activationspecification import ActivationSpecification
from quantlib.editing.editing.editors import Finder
from quantlib.editing.graphs.fx import FXOpcodeClasses


class ActivationFinder(Finder):

    def __init__(self, specification: ActivationSpecification):

        if not isinstance(specification, ActivationSpecification):
            raise TypeError

        super(ActivationFinder, self).__init__()
        self._specification = specification

    @property
    def specification(self) -> ActivationSpecification:
        return self._specification

    def find(self, g: fx.GraphModule) -> List[ActivationNode]:
        nonmodular_nodes = filter(lambda n: (n.op in FXOpcodeClasses.CALL_NONMODULAR.value), g.graph.nodes)
        nonmodular_targets = filter(lambda n: (n.target in self.specification.targets), nonmodular_nodes)
        return [ActivationNode(n) for n in nonmodular_targets]

    def check_aps_commutativity(self, aps: List[ActivationNode]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))
