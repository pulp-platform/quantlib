import torch.fx as fx
from torch.fx.experimental.optimization import extract_subgraph
from typing import List

from quantlib.utils import quantlib_err_header


def find_ancestors(n: fx.Node) -> List[fx.Node]:
    """Find the dependencies of a given node in a computational graph.

    This function performs a depth-first traversal of the computational graph
    to discover all the dependencies of the argument node.
    """

    nodes = []

    predecessors = list(n.all_input_nodes)
    if len(predecessors) == 0:
        pass
    else:
        for p in n.all_input_nodes:
            nodes.extend(find_ancestors(p))

    nodes.append(n)  # mark node as traversed

    return nodes


def extract_network_up_to(module_name: str, g: fx.GraphModule) -> fx.GraphModule:
    """Extract a sub-network terminating at the given ``nn.Module``.

    This function extracts an small ``nn.Module`` from a target ``nn.Module``.
    """

    if module_name not in dict(g.named_modules()).keys():
        raise ValueError(quantlib_err_header() + f"The target fx.GraphModule does not contain an nn.Module with name {module_name}.")

    n = next(iter([n for n in g.graph.nodes if n.target == module_name]))
    g = extract_subgraph(orig_module=g, nodes=find_ancestors(n), inputs=[], outputs=[n])
    g.recompile()

    return g
