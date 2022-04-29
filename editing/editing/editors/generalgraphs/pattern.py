import torch.nn as nn
import torch.fx as fx
import networkx as nx
from typing import Callable, Dict, List, Set

from .fx2nx import NXFXGraph
from quantlib.editing.graphs import FXOPCODES_IO, FXOPCODE_CALL_MODULE, FXOPCODE_CALL_METHOD, FXOPCODE_CALL_FUNCTION


def node_match_fx(pn:                    fx.Node,
                  dn:                    fx.Node,
                  pattern_gm:            fx.GraphModule,
                  data_gm:               fx.GraphModule,
                  pn_to_module_checkers: Dict[fx.Node, List[Callable[[nn.Module], bool]]]) -> bool:

    state: bool = True

    pn_opcode = pn.op
    dn_opcode = dn.op

    if pn_opcode in FXOPCODES_IO:  # I/O nodes act as wildcards
        state = True

    else:
        if pn_opcode != dn_opcode:
            state = False

        else:

            if pn_opcode in FXOPCODE_CALL_MODULE:
                pmodule = pattern_gm.get_submodule(target=pn.target)
                dmodule = data_gm.get_submodule(target=dn.target)
                cond_type = isinstance(dmodule, type(pmodule))
                cond_checkers = all(c(dmodule) for c in pn_to_module_checkers[pn]) if pn in pn_to_module_checkers.keys() else True
                state = cond_type & cond_checkers

            elif pn_opcode in FXOPCODE_CALL_METHOD:
                state = True if pn.target == dn.target else False

            elif pn_opcode in FXOPCODE_CALL_FUNCTION:
                state = True if pn.target.__name__ == dn.target.__name__ else False

            else:
                pass

    return state


class GenericGraphPattern(object):

    def __init__(self,
                 module:                  nn.Module,
                 symbolic_trace_fn:       Callable,
                 name_to_module_checkers: Dict[str, List[Callable[[nn.Module], bool]]]):

        super(GenericGraphPattern, self).__init__()

        self._module_names: Set[str] = set(dict(module.named_children()).keys())

        self._pattern_gm: fx.GraphModule = symbolic_trace_fn(root=module)
        self._pattern_nxg: NXFXGraph = NXFXGraph.from_fx_graph(self.pattern_fxg)

        self._pattern_node_to_module_checkers: Dict[fx.Node, List[Callable[[nn.Module], bool]]] = {self.name_to_pattern_node()[k]: v for k, v in name_to_module_checkers.items()}

    @property
    def pattern_gm(self) -> fx.GraphModule:
        return self._pattern_gm

    @property
    def pattern_fxg(self) -> fx.Graph:
        return self._pattern_gm.graph

    @property
    def pattern_nxg(self) -> nx.DiGraph:
        return self._pattern_nxg

    def name_to_pattern_node(self) -> Dict[str, fx.Node]:
        assert self._module_names.issubset(set(map(lambda n: n.target, self.pattern_fxg.nodes)))
        return {n.target: n for n in filter(lambda n: (n.op in FXOPCODE_CALL_MODULE) and (n.target in self._module_names), self.pattern_fxg.nodes)}

    def name_to_match_node(self, nodes_map: Dict[fx.Node, fx.Node]) -> Dict[str, fx.Node]:
        name_to_pattern_node = self.name_to_pattern_node()
        if not set(name_to_pattern_node.values()).issubset(set(nodes_map.keys())):
            raise RuntimeError  # I assume that those each pattern `fx.Node` which has been explicitly described has also been mapped to a corresponding data `fx.Node` during matching.
        return {k: nodes_map[v] for k, v in name_to_pattern_node.items()}

    def name_to_match_module(self, nodes_map: Dict[fx.Node, fx.Node], data_gm: fx.GraphModule) -> Dict[str, nn.Module]:
        name_to_match_node = self.name_to_match_node(nodes_map)
        target_to_module = dict(data_gm.named_modules())
        if not set(map(lambda v: v.target, name_to_match_node.values())).issubset(set(target_to_module.keys())):
            raise RuntimeError  # I assume that those data `fx.Node`s that have been matched against pattern `fx.Node`s with opcode `call_module` have themselves opcode `call_module`.
        return {k: target_to_module[v.target] for k, v in name_to_match_node.items()}

    def check_matched_module(self, pn: fx.Node, dn_module: nn.Module) -> bool:
        """Complement the semantic checks on the data graph at matching time."""
        return all(c(dn_module) for c in self._pattern_node_to_module_checkers[pn])

    def get_node_matching_function(self, data_gm: fx.GraphModule) -> Callable[[Dict, Dict], bool]:  # NetworkX nodes are implemented as dictionaries

        from functools import partial
        fn = partial(node_match_fx, **{'data_gm': data_gm, 'pattern_gm': self.pattern_gm, 'pn_to_module_checkers': self._pattern_node_to_module_checkers})

        def node_match_nx(dn: Dict, pn: Dict) -> bool:  # NetworkX nodes are implemented as dictionaries
            return fn(pn=pn['fx'], dn=dn['fx'])

        return node_match_nx
