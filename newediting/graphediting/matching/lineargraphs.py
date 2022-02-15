from collections import OrderedDict
import torch.nn as nn
import torch.fx as fx
from torch.fx.subgraph_rewriter import Match
from typing import List, Callable
import warnings

from quantlib.newediting.tracing import _OPCODE_PLACEHOLDER, _OPCODE_OUTPUT, _OPCODES_IO, _OPCODE_CALL_MODULE
from quantlib.newutils import quantlib_err_header, quantlib_wng_header


class LinearGraphMatcher(object):

    def __init__(self,
                 symbolic_trace_fn: Callable,
                 pattern: nn.Sequential):
        """An object to identify linear sub-graphs into a target graph.

        **Linear graphs** are amongst the simplest directed graphs. A linear
        graph is a graph :math:`G = (V_{G}, E_{G})` such that there exists an
        enumeration :math:`(v_{0}, \dots, v_{N - 1})` of :math:`V_{G}` (where
        :math:`N = |V_{G}| \geq 1`) according to which :math:`E_{G} = \{
        (v_{i}, v_{i+1}), i = 0, \dots, N - 2 \}`. In other words, a linear
        graph is an acyclic graph such that:

        * there is a single node :math:`v_{0}` which has exactly zero inbound
          nodes and exactly one outbound node (the *entry point*);
        * there is a single node :math:`v_{N - 1}` which has exactly one
          inbound node and exactly zero outbound nodes (the *exit point*);
        * all the other nodes have exactly one inbound and one outbound node.

        Despite their simplicity, these graphs represent a wide range of
        sub-graphs used in computational graphs. In PyTorch, linear graphs are
        described using the ``nn.Sequential`` construct, which executes the
        ``nn.Module``s passed to its constructor one after the other.

        **Singleton graphs** are a special sub-case of linear graphs. Indeed,
        singleton graphs contain a single vertex :math:`v_{0}` and have no
        arcs, i.e., :math:`E_{G} = \emptyset`.

        Note that ``torch.fx`` tracing mechanism adds a placeholder node in
        front of :math:`v_{0}` and an output node after :math:`v_{N - 1}`,
        effectively turning a linear graph with :math:`N` nodes into a linear
        graph with :math:`N + 2` nodes. Therefore, this class supports pattern
        linear graphs which have at least three nodes.

        """

        self._symbolic_trace_fn = symbolic_trace_fn

        # trace the pattern `nn.Module`
        self._pattern_gm: fx.GraphModule = self._symbolic_trace_fn(root=pattern)
        self._pattern: fx.Graph = self._pattern_gm.graph
        if not LinearGraphMatcher._is_linear_graph(self._pattern):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "requires a path graph as template.")
        self._pattern_anchor = next(iter(reversed(self._pattern.nodes)))

        # decompose the nodes of the template part (L-term) of the graph rewriting rule
        self._Kterm_entry = tuple(pn for pn in self._pattern.nodes if pn.op in _OPCODE_PLACEHOLDER)
        self._Kterm_exit  = tuple(pn for pn in self._pattern.nodes if pn.op in _OPCODE_OUTPUT)
        self._Lterm_core  = tuple(pn for pn in self._pattern.nodes if (pn not in self._Kterm_entry) and (pn not in self._Kterm_exit))

    @staticmethod
    def _is_entry_node(n: fx.Node) -> bool:
        cond_in  = len(n.all_input_nodes) == 0
        cond_out = len(n.users) == 1            # has exactly one output
        return cond_in & cond_out

    @staticmethod
    def _is_body_node(n: fx.Node) -> bool:
        cond_in  = len(n.all_input_nodes) == 1  # has exactly one input
        cond_out = len(n.users) == 1            # has exactly one output
        return cond_in & cond_out

    @staticmethod
    def _is_exit_node(n: fx.Node) -> bool:
        cond_in  = len(n.all_input_nodes) == 1  # has exactly one input
        cond_out = len(n.users) == 0
        return cond_in & cond_out

    @staticmethod
    def _is_linear_graph(g: fx.Graph) -> bool:

        state = True

        for n in g.nodes:

            if n.op in _OPCODE_PLACEHOLDER:
                state &= LinearGraphMatcher._is_entry_node(n)
            elif n.op in _OPCODE_OUTPUT:
                state &= LinearGraphMatcher._is_exit_node(n)
            else:
                state &= LinearGraphMatcher._is_body_node(n)

            if state is False:
                break

        return state

    @staticmethod
    def _check_node_attributes(pn: fx.Node, dn: fx.Node, pattern_gm: fx.GraphModule, data_gm: fx.GraphModule) -> bool:
        """Compare the attributes of two nodes.

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
          the attributes of :math:`L` filters the candidate matches to return
          those which have attributes compatible with the prescriptions of the
          pattern :math:`L`.

        Instead of solving the problem in this two-step fashion, we follow the
        style of ``torch.fx`` and interleave attribute checking with the
        solution of the graph isomorphism problem.

        """

        if pn.op in _OPCODES_IO:
            state = True

        elif pn.op in _OPCODE_CALL_MODULE:
            # The `torch.fx` implementation of `SubgraphMatcher` check
            # that the `target` attributes of candidate matching nodes are
            # the same. We are instead interested in determining whether
            # the two nodes represent instances of the same `nn.Module` class.
            cond_op = pn.op == dn.op
            try:
                pn_module_class  = type(dict(pattern_gm.named_modules())[pn.target])
                dn_module_object = dict(data_gm.named_modules())[dn.target]
                cond_type = isinstance(dn_module_object, pn_module_class)
            except KeyError:
                cond_type = False
            state = cond_op & cond_type

        else:
            cond_op     = pn.op == dn.op
            cond_target = pn.target == dn.target  # e.g., `call_function` nodes take the name of the underlying `torch` function
            state = cond_op & cond_target

        return state

    @staticmethod
    def _match_nodes(pn: fx.Node,
                     dn: fx.Node,
                     pattern_gm: fx.GraphModule,
                     data_gm: fx.GraphModule) -> List[OrderedDict]:
        """The recursive back-tracking method."""

        matches: List[OrderedDict] = []

        # TODO: When using non-linear pattern graphs there might be branching
        #       points. Such condition makes the traversal of the data graph
        #       more delicate. In fact, we would need to check that for a
        #       given traversal we never assign two different "roles" to the
        #       same node of the data graph (i.e., that this node is matched
        #       with two different nodes in the pattern graph).

        if not LinearGraphMatcher._check_node_attributes(pn, dn, pattern_gm, data_gm):
            pass  # node signatures (i.e., node attributes) do not match: back-track

        else:

            if pn.op in _OPCODE_PLACEHOLDER:  # terminal condition
                partial_match = OrderedDict([(pn, dn)])
                matches.append(partial_match)

            elif pn.op in _OPCODE_OUTPUT:

                if len(dn.all_input_nodes) == 0:  # it is impossible to continue the matching along the data graph: back-track
                    pass

                else:
                    partial_match = OrderedDict([(pn, dn)])
                    pn_ = next(iter(pn.all_input_nodes))
                    partial_matches = []
                    for dn_ in dn.all_input_nodes:
                        partial_matches += LinearGraphMatcher._match_nodes(pn_, dn_, pattern_gm, data_gm)

                    if len(partial_matches) > 0:  # assemble partial candidate solution (PCS)
                        for pm in partial_matches:
                            map_ = partial_match.copy()
                            map_.update(**pm)
                            matches.append(map_)
                    else:  # no matches were found: back-track
                        pass

            else:  # we are matching the "core" nodes of the pattern

                if not LinearGraphMatcher._is_body_node(dn):  # this node can't be modified, since it might be used in scopes other than the one defined by the pattern: back-track
                    pass

                else:
                    partial_match = OrderedDict([(pn, dn)])
                    partial_matches = LinearGraphMatcher._match_nodes(next(iter(pn.all_input_nodes)), next(iter(dn.all_input_nodes)), pattern_gm, data_gm)

                    if len(partial_matches) > 0:  # assemble partial candidate solution (PCS)
                        for pm in partial_matches:
                            map_ = partial_match.copy()
                            map_.update(**pm)
                            matches.append(map_)
                    else:  # no matches were found: back-track
                        pass

        return matches

    def find_application_points(self, data_gm: fx.GraphModule) -> List[Match]:
        """Find the sub-graphs of the data graph matching the pattern graph.

        This function uses back-tracking to identify all the linear sub-graphs
        of ``data_gm`` that match ``self.pattern_gm``.
        """
        # ensure that the data graph IR is the same as the pattern graph's
        data_gm = self._symbolic_trace_fn(root=data_gm)

        matched_nodes = set()

        def overlaps_with_previous_matches(match: Match) -> bool:
            body_nodes = set(dn for pn, dn in match.nodes_map.items() if (dn.op not in _OPCODES_IO))
            return any(n in matched_nodes for n in body_nodes)

        matches: List[Match] = []
        for data_anchor in reversed(data_gm.graph.nodes):

            candidate_matches = LinearGraphMatcher._match_nodes(self._pattern_anchor, data_anchor, self._pattern_gm, data_gm)
            for candidate_match in candidate_matches:
                match = Match(anchor=data_anchor, nodes_map=candidate_match)
                if not overlaps_with_previous_matches(match):
                    matches.append(match)
                    matched_nodes.union(set(dn for pn, dn in match.nodes_map.items() if pn not in _OPCODES_IO))
                else:
                    warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + "two matches with non-zero overlap were found; the latest will be discarded.")

        return matches
