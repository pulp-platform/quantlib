from functools import partial
import torch.nn as nn
from typing import Type, Tuple, List

from .node import Node
from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule


def custom_traverse(module:         nn.Module,
                    qualified_name: str = '',
                    leaf_types:     Tuple[Type[nn.Module]] = tuple()) -> List[Node]:
    """Traverse the target ``nn.Module`` tree down to custom leaves or atoms.

    In some scenarios, users might want to identify container ``nn.Module``s
    as opposed to atomic ``nn.Module``s. Containers could be traversed
    further down, but we prune the traversal if their type matches one of the
    specified ones.
    """
    nodes = []

    if isinstance(module, leaf_types) or len(list(module.named_children())) == 0:
        nodes.append(Node(name=qualified_name, module=module))

    else:
        for child_name, child_module in module.named_children():
            child_qualified_name = child_name if qualified_name == '' else '.'.join([qualified_name, child_name])
            nodes.extend(custom_traverse(module=child_module, qualified_name=child_qualified_name, leaf_types=leaf_types))

    return nodes


qmodule_traverse = partial(custom_traverse, leaf_types=(_QModule,))
