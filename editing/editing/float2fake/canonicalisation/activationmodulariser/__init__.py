from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activationspecification import NonModularTargets, ActivationSpecification
from .finder import ActivationFinder
from .applier import ActivationReplacer
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from quantlib.editing.editing.editors import ComposedEditor


# describe all activation nodes that need to be canonicalised...
specifications = (
    ActivationSpecification(nn.ReLU,      NonModularTargets((torch.relu, torch.relu_, F.relu, F.relu_,), tuple())),
    ActivationSpecification(nn.ReLU6,     NonModularTargets((F.relu6,),                                  tuple())),
    ActivationSpecification(nn.LeakyReLU, NonModularTargets((F.leaky_relu, F.leaky_relu_,),              tuple())),
)

# ... then programmatically create each `Rewriter`
namespace = OrderedDict([])
for spec in specifications:

    # create the class
    class_name = spec.module_class.__name__ + 'Modulariser'

    def __init__(self_):
        Rewriter.__init__(self_, class_name, quantlib_symbolic_trace, ActivationFinder(spec), ActivationReplacer(spec))

    class_ = type(spec.module_class.__name__ + 'Modulariser', (Rewriter,), {'__init__': __init__})

    # add the class to this module's namespace and to the namespace used to build `ActivationModulariser`
    globals()[class_name] = class_
    namespace[class_name] = class_


# create the general-purpose `ActivationModulariser`
class ActivationModulariser(ComposedEditor):
    def __init__(self):
        super(ActivationModulariser, self).__init__([class_() for class_ in namespace.values()])


__all__ = [n for n in namespace.keys()] + ['ActivationModulariser']
