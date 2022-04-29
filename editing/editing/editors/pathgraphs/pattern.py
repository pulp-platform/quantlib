"""
This module implements the functionalities to describe path graphs using
PyTorch's modular API (i.e., ``nn.Sequential`` objects) in such a way to
facilitate the creation of ``Rewriter``s that act on such graphs. In
particular, these descriptions will include the following properties:

* symbolic names, to facilitate access to the different ``nn.Module``s that
  compose the pattern graph;
* checker functions, to verify that the composing ``nn.Module``s are such that
  they actually require treatment by the ``Rewriter`` (e.g., that a linear
  ``nn.Module`` followed by a batch normalisation has a bias, which can thus
  be folded into the batch normalisation itself).

"""

from collections import OrderedDict
import torch.nn as nn
import torch.fx as fx
from typing import Callable, List, Union, Optional, NamedTuple, Tuple, Dict, Set

from quantlib.editing.graphs import FXOPCODE_CALL_MODULE


# -- MODULE-LEVEL DESCRIPTION -- #

ModuleChecker = Callable[[nn.Module], bool]


class ModuleDescription(NamedTuple):
    name:     str
    module:   nn.Module
    checkers: Optional[Union[ModuleChecker, List[ModuleChecker]]] = None


def canonicalise_module_description(md: ModuleDescription) -> ModuleDescription:

    # the only component that needs canonicalisation is actually the `checkers`
    if md.checkers is None:
        checkers = []
    elif isinstance(md.checkers, Callable):  # is it a `ModuleChecker`?
        checkers = [md.checkers]
    elif isinstance(md.checkers, list):
        assert all(map(lambda c: isinstance(c, Callable), md.checkers))  # is it a list of `ModuleChecker`s?
        checkers = md.checkers
    else:
        raise TypeError

    return ModuleDescription(name=md.name, module=md.module, checkers=checkers)


def verify_module_description(md: ModuleDescription) -> bool:
    return all(c(md.module) for c in md.checkers)


# -- GRAPH-LEVEL DESCRIPTION -- #

PathGraphDescription = Tuple[ModuleDescription, ...]


def canonicalise_path_graph_description(pgd: PathGraphDescription) -> PathGraphDescription:
    """Canonicalise each pattern component in the given graph description."""
    return tuple(map(lambda md: canonicalise_module_description(md), pgd))


def verify_path_graph_description(pgd: PathGraphDescription) -> bool:
    """Verify that the pattern components satisfy the constraints set by the
    corresponding ``ModuleChecker``s.
    """
    return all(verify_module_description(md) for md in pgd)


def validate_path_graph_description(pgd: PathGraphDescription) -> PathGraphDescription:
    """Canonicalise the given graph description, then verify that the pattern
    items satisfy the constraints set by the corresponding ``ModuleChecker``s.
    """

    pgd = canonicalise_path_graph_description(pgd)

    if verify_path_graph_description(pgd):
        return pgd
    else:
        raise RuntimeError  # TODO: indicate which components are wrong


# -- PATTERN OBJECT -- #

class PathGraphPattern(object):

    def __init__(self,
                 pgd:               PathGraphDescription,
                 symbolic_trace_fn: Callable):

        super(PathGraphPattern, self).__init__()

        pgd = validate_path_graph_description(pgd)
        self._module_names = {md.name for md in pgd}

        # create the pattern `fx.GraphModule`
        pattern_module: nn.Sequential = nn.Sequential(OrderedDict([(md.name, md.module) for md in pgd]))  # turn the given graph description into an ``nn.Sequential`` object
        self._pattern_gm: fx.GraphModule = symbolic_trace_fn(root=pattern_module)
        self._leakable_nodes: Set[fx.Node] = set()

        # map nodes of the pattern `fx.Graph` to the corresponding `nn.Module` checks
        name_to_module_checkers: Dict[str, List[Callable[[nn.Module], bool]]] = OrderedDict([(md.name, md.checkers) for md in pgd])  # extract ``ModuleChecker`` lists from the given graph description
        self._pattern_node_to_module_checkers: Dict[fx.Node, List[Callable[[nn.Module], bool]]] = {self.name_to_pattern_node()[k]: v for k, v in name_to_module_checkers.items()}

    @property
    def pattern_gm(self) -> fx.GraphModule:
        return self._pattern_gm

    @property
    def pattern_g(self) -> fx.Graph:
        return self._pattern_gm.graph

    def set_leakable_nodes(self, nodes: Union[fx.Node, Set[fx.Node]]) -> None:
        """It might be not relevant that some nodes in the pattern are used by
        ``fx.Node``s outside the match, since their outputs will remain
        unchanged even after the rewriting.
        """

        # canonicalise inputs
        if isinstance(nodes, fx.Node):
            nodes = {nodes}
        elif isinstance(nodes, set):
            assert all(isinstance(n, fx.Node) for n in nodes)
            pass
        else:
            raise TypeError

        # check input validity
        if any(n not in self.pattern_g.nodes for n in nodes):
            raise ValueError

        self._leakable_nodes = self._leakable_nodes.union(nodes)

    @property
    def leakable_nodes(self) -> Set[fx.Node]:
        return self._leakable_nodes

    def name_to_pattern_node(self) -> Dict[str, fx.Node]:
        assert self._module_names.issubset(set(map(lambda n: n.target, self.pattern_g.nodes)))
        return {n.target: n for n in filter(lambda n: (n.op in FXOPCODE_CALL_MODULE) and (n.target in self._module_names), self.pattern_g.nodes)}

    def name_to_match_node(self, nodes_map: Dict[fx.Node, fx.Node]) -> Dict[str, fx.Node]:
        name_to_pattern_node = self.name_to_pattern_node()
        if not set(name_to_pattern_node.values()).issubset(set(nodes_map.keys())):
            raise RuntimeError  # I assume that those each pattern `fx.Node` which has been explicitly described has also been mapped to a corresponding data `fx.Node` during matching.
        return {k: nodes_map[v] for k, v in name_to_pattern_node.items()}

    def name_to_match_module(self, nodes_map: Dict[fx.Node, fx.Node], data_gm: fx.GraphModule) -> Dict[str, nn.Module]:
        name_to_match_node = self.name_to_match_node(nodes_map)
        target_to_module = dict(data_gm.named_modules())
        if not set(map(lambda v: v.target, name_to_match_node.values())).issubset(set(target_to_module.keys())):
            A = set(map(lambda v: v.target, name_to_match_node.values()))
            B = set(target_to_module.keys())
            print(A.difference(B))
            raise RuntimeError  # I assume that those data `fx.Node`s that have been matched against pattern `fx.Node`s with opcode `call_module` have themselves opcode `call_module`.
        return {k: target_to_module[v.target] for k, v in name_to_match_node.items()}

    def check_matched_module(self, pn: fx.Node, dn_module: nn.Module) -> bool:
        """Complement the semantic checks on the data graph at matching time."""
        return all(c(dn_module) for c in self._pattern_node_to_module_checkers[pn])
