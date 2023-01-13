# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from __future__ import annotations

import torch.nn as nn
import torch.fx as fx
from typing import Tuple, List, Dict

from ...applicationpoint import NodesMap
from .nnmodulewithcheckers import Checker, NNModuleWithCheckers, NNModuleWithCheckersSpecType, resolve_nnmodulewithcheckersspec
from quantlib.editing.graphs.fx import SymbolicTraceFnType
from quantlib.editing.graphs.fx import FXOpcodeClasses


class NNModulePattern(object):
    """Base class abstracting a traced ``nn.Module`` with rich semantic checks."""

    def __init__(self,
                 symbolic_trace_fn:      SymbolicTraceFnType,
                 modulewithcheckersspec: NNModuleWithCheckersSpecType):

        self._symbolic_trace_fn = symbolic_trace_fn  # `Rewriter`s built around this pattern should assume that their data graphs have been traced with this function

        # trace the `nn.Module`
        module_with_checkers: NNModuleWithCheckers = resolve_nnmodulewithcheckersspec(modulewithcheckersspec)
        module_with_checkers.module.eval()  # Note that we trace the validation version of the computational graph: keep this in mind when running pattern matching on data graphs that are in training state.
        self._gm = self._symbolic_trace_fn(root=module_with_checkers.module)

        # pull-back the map from `nn.Module`s to `ModuleChecker`s to a map from `fx.Node`s to `ModuleChecker`s
        name_to_pattern_node = self.name_to_pattern_node()
        assert set(name_to_pattern_node.keys()).issubset(set(module_with_checkers.name_to_checkers.keys()))
        self._node_to_checkers: Dict[fx.Node, Tuple[Checker, ...]] = {node: module_with_checkers.name_to_checkers[name] for name, node in name_to_pattern_node.items()}

    @property
    def symbolic_trace_fn(self) -> SymbolicTraceFnType:
        return self._symbolic_trace_fn

    @property
    def gm(self) -> fx.GraphModule:
        return self._gm

    @property
    def fxg(self) -> fx.Graph:
        return self.gm.graph

    @property
    def fxg_module_nodes(self) -> List[fx.Node]:
        return list(filter(lambda n: (n.op in FXOpcodeClasses.CALL_MODULE.value) and (n.target in set(dict(self.gm.named_children()).keys())), self.fxg.nodes))

    @property
    def node_to_checkers(self) -> Dict[fx.Node, Tuple[Checker, ...]]:
        return self._node_to_checkers

    def name_to_pattern_node(self) -> Dict[str, fx.Node]:
        return {n.target: n for n in self.fxg_module_nodes}

    def name_to_match_node(self, nodes_map: NodesMap) -> Dict[str, fx.Node]:
        name_to_pattern_node = self.name_to_pattern_node()
        if not set(name_to_pattern_node.values()).issubset(set(nodes_map.keys())):
            raise RuntimeError  # I assume that each `fx.Node` in the pattern has been mapped to a corresponding data `fx.Node` during matching.
        return {k: nodes_map[v] for k, v in name_to_pattern_node.items()}

    def name_to_match_module(self, nodes_map: NodesMap, data_gm: fx.GraphModule) -> Dict[str, nn.Module]:
        name_to_match_node = self.name_to_match_node(nodes_map)
        match_node_to_match_target = {n: n.target for n in name_to_match_node.values()}
        match_target_to_match_module = dict(data_gm.named_modules())
        if not set(match_node_to_match_target.values()).issubset(set(match_target_to_match_module.keys())):
            raise RuntimeError  # I assume that each `fx.Node` in the match that have been matched against pattern `fx.Node`s with opcode `call_module` have themselves opcode `call_module`.
        return {k: match_target_to_match_module[match_node_to_match_target[v]] for k, v in name_to_match_node.items()}

    @staticmethod
    def check_node_attributes(pattern:    NNModulePattern,
                              pn:         fx.Node,
                              dn:         fx.Node,
                              data_gm:    fx.GraphModule) -> bool:
        """Compare the semantic attributes of two nodes.

        Given a graph rewriting rule :math:`\rho = (K, L, R)`, a sub-graph
        :math:`H` of the target graph :math:`G` is a **match** for the pattern
        :math:`L` if two conditions are satisfied:

        * :math:`H` is isomorphic to :math:`L`;
        * the attributes of :math:`H` are compatible with those of :math:`L`.

        :math:`H` is called either a match for :math:`L`, or an **application
        point** for :math:`\rho`. According to the two conditions above,
        discovering application points requires solving two sub-problems:

        * solving the sub-graph isomorphism problem returns candidate
          sub-graphs :math:`H \subseteq G` which are isomorphic to :math:`L`;
        * performing signature matching over the attributes of :math:`H` and
          the attributes of :math:`L` nametomodule the candidate matches to return
          those which have attributes compatible with the prescriptions of the
          pattern :math:`L`.

        Instead of solving the problem in this two-step fashion, we follow the
        style of ``torch.fx`` and interleave attribute checking with the
        solution of the graph isomorphism problem.

        """

        if pn.op in FXOpcodeClasses.IO.value:  # I/O nodes act as wildcards
            state = True

        else:

            if pn.op != dn.op:
                state = False

            else:
                if pn.op in FXOpcodeClasses.CALL_MODULE.value:
                    dmodule = data_gm.get_submodule(target=dn.target)
                    # verify whether the candidate matched module has the functionality of the corresponding pattern module
                    state = all(c(dmodule) for c in pattern.node_to_checkers[pn])

                elif pn.op in FXOpcodeClasses.CALL_METHOD.value:
                    state = True if (pn.target == dn.target) else False

                elif pn.op in FXOpcodeClasses.CALL_FUNCTION.value:
                    state = True if (pn.target.__name__ == dn.target.__name__) else False

                else:
                    assert pn.op in FXOpcodeClasses.GET_ATTR.value
                    raise NotImplementedError

        return state
