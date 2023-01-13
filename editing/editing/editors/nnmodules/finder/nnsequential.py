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

import warnings
from collections import OrderedDict
import torch.fx as fx
from typing import List, Set, Dict

from ..applicationpoint import NodesMap
from ..pattern import NNSequentialPattern
from .base import NNModuleMatcher
from quantlib.editing.graphs.fx import FXOpcodeClasses
from quantlib.utils.messages import quantlib_wng_header


class PathGraphMatcher(NNModuleMatcher):

    def __init__(self, pattern: NNSequentialPattern):
        """An object to identify path sub-graphs into a target graph.

        **Path graphs** are amongst the simplest directed graphs. A path graph
        is a graph :math:`G = (V_{G}, E_{G})` such that there exists an
        enumeration :math:`(v_{0}, \dots, v_{N - 1})` of :math:`V_{G}` (where
        :math:`N = |V_{G}| > 1`) according to which :math:`E_{G} = \{ (v_{i},
        v_{i+1}), i = 0, \dots, N - 2 \}`. In other words, a path graph is an
        acyclic graph such that:

        * there is a single node :math:`v_{0}` which has exactly zero inbound
          nodes and exactly one outbound node (the *entry point*);
        * there is a single node :math:`v_{N - 1}` which has exactly one
          inbound node and exactly zero outbound nodes (the *exit point*);
        * all the other nodes have exactly one inbound and one outbound node.

        Despite their simplicity, these graphs represent a wide range of
        sub-graphs used in computational graphs. In PyTorch, path graphs are
        described using the ``nn.Sequential`` construct, which executes the
        ``nn.Module``s passed to its constructor one after the other.

        **Singleton graphs** are a special sub-case of path graphs. Indeed,
        singleton graphs contain a single vertex :math:`v_{0}` and have no
        arcs, i.e., :math:`E_{G} = \emptyset`.

        Note that ``torch.fx`` tracing mechanism adds a placeholder node in
        front of :math:`v_{0}` and an output node after :math:`v_{N - 1}`,
        effectively turning a path graph with :math:`N` nodes into a linear
        graph with :math:`N + 2` nodes. Therefore, this class supports pattern
        path graphs which have at least three nodes.

        """

        if not isinstance(pattern, NNSequentialPattern):
            raise TypeError
        if not PathGraphMatcher.is_path_graph(pattern.fxg):
            raise ValueError

        super(PathGraphMatcher, self).__init__(pattern)
        self._pattern_anchor = next(iter(reversed(self.pattern.fxg.nodes)))

        # # decompose the nodes of the pattern/template part (L-term) of the graph rewriting rule
        # self._Kterm_entry = tuple(filter(lambda pn: pn.op in FXOPCODE_PLACEHOLDER, self.pattern.pattern_g.nodes))
        # self._Kterm_exit  = tuple(filter(lambda pn: pn.op in FXOPCODE_OUTPUT, self.pattern.pattern_g.nodes))
        # self._Lterm_body  = tuple(filter(lambda pn: (pn not in self._Kterm_entry) and (pn not in self._Kterm_exit), self.pattern.pattern_g.nodes))

    # -- TOPOLOGICAL CHECKERS -- #

    @staticmethod
    def is_entry_node(n: fx.Node) -> bool:
        cond_in  = len(n.all_input_nodes) == 0  # has exactly zero inputs
        cond_out = len(n.users) == 1            # has exactly one output
        return cond_in & cond_out

    @staticmethod
    def is_exit_node(n: fx.Node) -> bool:
        cond_in  = len(n.all_input_nodes) == 1  # has exactly one input
        cond_out = len(n.users) == 0            # has exactly zero outputs
        return cond_in & cond_out

    @staticmethod
    def is_body_node(n: fx.Node) -> bool:
        cond_in  = len(n.all_input_nodes) == 1  # has exactly one input
        cond_out = len(n.users) == 1            # has exactly one output
        return cond_in & cond_out

    @staticmethod
    def is_path_graph(g: fx.Graph) -> bool:

        state: bool = True

        for n in g.nodes:

            if n.op in FXOpcodeClasses.PLACEHOLDER.value:
                state &= PathGraphMatcher.is_entry_node(n)
            elif n.op in FXOpcodeClasses.OUTPUT.value:
                state &= PathGraphMatcher.is_exit_node(n)
            else:
                state &= PathGraphMatcher.is_body_node(n)

            if state is False:
                break

        return state

    # -- PATTERN MATCHING -- #

    @staticmethod
    def _assemble_pcs(matches: List[NodesMap], prefix: NodesMap, postfixes: List[NodesMap]) -> List[NodesMap]:
        """Assemble partial candidate solution (PCS).

        This function works by side-effect on ``matches``.
        """
        if len(postfixes) > 0:
            for postfix in postfixes:
                match = prefix.copy()
                match.update(**postfix)
                matches.append(match)
        else:  # no matches were found: back-track
            pass

    @staticmethod
    def _match_nodes(pattern:    NNSequentialPattern,
                     pn:         fx.Node,
                     dn:         fx.Node,
                     data_gm:    fx.GraphModule) -> List[NodesMap]:
        """Solve sub-graph isomorphism using back-tracking."""

        matches: List[NodesMap] = []

        # TODO: When using non-path pattern graphs there might be branching
        #       points. Such condition makes the traversal of the data graph
        #       more delicate. In fact, we would need to check that for a
        #       given traversal we never assign two different "roles" to the
        #       same node of the data graph (i.e., that this node is matched
        #       with two different nodes in the pattern graph).

        if not pattern.check_node_attributes(pattern, pn, dn, data_gm):
            pass  # node semantics (i.e., node attributes) do not match: back-track

        else:

            if pn.op in FXOpcodeClasses.PLACEHOLDER.value:  # terminal condition
                match = NodesMap([(pn, dn)])
                matches.append(match)

            elif pn.op in FXOpcodeClasses.OUTPUT.value:

                if len(dn.all_input_nodes) == 0:  # it is impossible to continue the matching along the data graph: back-track
                    pass

                else:
                    match_prefix = NodesMap([(pn, dn)])       # this is the prefix of all the matches rooted at `dn`
                    match_postfixes = []                      # here we will store the postfixes of all the matches rooted at `dn`
                    next_pn = next(iter(pn.all_input_nodes))  # traverse the pattern upstream
                    for next_dn in dn.all_input_nodes:        # traverse the data graph upstream
                        match_postfixes += PathGraphMatcher._match_nodes(pattern, next_pn, next_dn, data_gm)
                    PathGraphMatcher._assemble_pcs(matches, match_prefix, match_postfixes)

            else:  # we are matching the "core" nodes of the pattern

                if not (PathGraphMatcher.is_body_node(dn) or (pn in pattern.leakable_nodes)):  # this node can't be modified, since it might be used in scopes other than the one defined by the pattern: back-track
                    pass

                else:
                    match_prefix = NodesMap([(pn, dn)])
                    match_postfixes = PathGraphMatcher._match_nodes(pattern, next(iter(pn.all_input_nodes)), next(iter(dn.all_input_nodes)), data_gm)
                    PathGraphMatcher._assemble_pcs(matches, match_prefix, match_postfixes)

        return matches

    # @staticmethod
    # def _overlaps_with_previous_matches(matched_nodes: Set[fx.Node], match: NodesMap) -> bool:
    #     body_nodes = set(dn for pn, dn in match.items() if (pn.op not in FXOpcodeClasses.IO.value))
    #     return any(n in matched_nodes for n in body_nodes)

    @staticmethod
    def _overlaps_with_previous_matches(matched_nodes: Dict[fx.Node, Set[fx.Node]], match: NodesMap) -> bool:
        temp = OrderedDict([(pn, {dn}) for pn, dn in match.items() if (pn in matched_nodes.keys())])
        if all((len(matched_nodes[pn].intersection(temp[pn])) == 0) for pn in matched_nodes.keys()):
            state = False
            for pn in matched_nodes.keys():
                matched_nodes[pn] = matched_nodes[pn].union(temp[pn])
        else:
            state = True
        return state

    def find(self, data_gm: fx.GraphModule) -> List[NodesMap]:
        """Find the sub-graphs of the data graph matching the pattern graph.

        Note that this function makes the following assumption: that the IR of
        the data graph (i.e., its ``fx.Graph``) has been obtained using the
        same tracing function used to generate the query graph (i.e., the
        ``symbolic_trace_fn`` passed to the constructor of ``self._pattern``).

        """

        # TODO: ensure that `data_gm == symbolic_trace_fn(root=data_gm)`,
        #       i.e., that the data graph has been obtained using the same
        #       tracing function as the pattern (query, template) graph.

        matches: List[NodesMap] = []

        # matched_nodes: Set[fx.Node] = set()
        matched_nodes: Dict[fx.Node, Set[fx.Node]] = OrderedDict([(pn, set()) for pn in self.pattern.fxg.nodes if not ((pn.op in FXOpcodeClasses.IO.value) or (pn in self.pattern.leakable_nodes))])

        for data_anchor in reversed(data_gm.graph.nodes):

            # find all matches rooted at the anchor
            candidate_matches = PathGraphMatcher._match_nodes(self.pattern, self._pattern_anchor, data_anchor, data_gm)

            # filter duplicate matchings
            for candidate_match in candidate_matches:

                if not PathGraphMatcher._overlaps_with_previous_matches(matched_nodes, candidate_match):
                    match = candidate_match
                    matches.append(match)
                    # body_nodes    = set(dn for pn, dn in match.items() if (pn.op not in FXOpcodeClasses.IO.value))
                    # matched_nodes = matched_nodes.union(body_nodes)
                else:
                    warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + "two matches with non-zero overlap were found; the most recently discovered will be discarded.")

        return matches

    def check_aps_commutativity(self, aps: List[NodesMap]) -> bool:
        return True  # TODO: implement the check!
