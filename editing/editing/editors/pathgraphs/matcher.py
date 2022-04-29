from collections import OrderedDict
import torch.fx as fx
from torch.fx.subgraph_rewriter import Match
from typing import List
import warnings

from quantlib.editing.editing.editors.pathgraphs import PathGraphPattern
from quantlib.editing.graphs import FXOPCODE_PLACEHOLDER, FXOPCODE_OUTPUT, FXOPCODES_IO, FXOPCODE_CALL_MODULE
from quantlib.utils import quantlib_err_header, quantlib_wng_header


class PathGraphMatcher(object):

    def __init__(self, pattern: PathGraphPattern):
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

        self._pattern = pattern
        if not PathGraphMatcher.is_path_graph(self.pattern.pattern_g):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "requires a path graph as pattern graph.")
        self._pattern_anchor = next(iter(reversed(self.pattern.pattern_g.nodes)))

        # # decompose the nodes of the pattern/template part (L-term) of the graph rewriting rule
        # self._Kterm_entry = tuple(filter(lambda pn: pn.op in FXOPCODE_PLACEHOLDER, self.pattern.pattern_g.nodes))
        # self._Kterm_exit  = tuple(filter(lambda pn: pn.op in FXOPCODE_OUTPUT, self.pattern.pattern_g.nodes))
        # self._Lterm_body  = tuple(filter(lambda pn: (pn not in self._Kterm_entry) and (pn not in self._Kterm_exit), self.pattern.pattern_g.nodes))

    @property
    def pattern(self) -> PathGraphPattern:
        return self._pattern

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

            if n.op in FXOPCODE_PLACEHOLDER:
                state &= PathGraphMatcher.is_entry_node(n)
            elif n.op in FXOPCODE_OUTPUT:
                state &= PathGraphMatcher.is_exit_node(n)
            else:
                state &= PathGraphMatcher.is_body_node(n)

            if state is False:
                break

        return state

    # -- PATTERN MATCHING -- #

    @staticmethod
    def _check_node_attributes(pattern:    PathGraphPattern,
                               pn:         fx.Node,
                               dn:         fx.Node,
                               pattern_gm: fx.GraphModule,
                               data_gm:    fx.GraphModule) -> bool:
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

        if pn.op in FXOPCODES_IO:
            state = True  # I/O `fx.Node`s are wildcards

        elif pn.op in FXOPCODE_CALL_MODULE:
            # The `torch.fx` implementation of `SubgraphMatcher` check
            # that the `target` attributes of candidate matching nodes are
            # the same. We are instead interested in determining whether
            # the two nodes represent instances of the same `nn.Module` class.
            cond_op = pn.op == dn.op
            try:
                pn_module   = pattern_gm.get_submodule(target=pn.target)
                dn_module   = data_gm.get_submodule(target=dn.target)
                cond_type   = isinstance(dn_module, type(pn_module))  # Is the data `nn.Module` a sub-class of the query `nn.Module`?
                cond_module = pattern.check_matched_module(pn, dn_module) if cond_type else False
            except AttributeError:
                cond_type   = False
                cond_module = False
            state = cond_op & cond_type & cond_module

        else:
            cond_op     = pn.op == dn.op
            cond_target = pn.target == dn.target  # e.g., `call_function` nodes take the name of the underlying `torch` function
            state = cond_op & cond_target

        return state

    @staticmethod
    def _match_nodes(pattern:    PathGraphPattern,
                     pn:         fx.Node,
                     dn:         fx.Node,
                     pattern_gm: fx.GraphModule,
                     data_gm:    fx.GraphModule) -> List[OrderedDict]:
        """Solve sub-graph isomorphism using back-tracking.

        This function is recursive.
        """

        matches: List[OrderedDict] = []

        # TODO: When using non-linear pattern graphs there might be branching
        #       points. Such condition makes the traversal of the data graph
        #       more delicate. In fact, we would need to check that for a
        #       given traversal we never assign two different "roles" to the
        #       same node of the data graph (i.e., that this node is matched
        #       with two different nodes in the pattern graph).

        if not PathGraphMatcher._check_node_attributes(pattern, pn, dn, pattern_gm, data_gm):
            pass  # node signatures (i.e., node attributes) do not match: back-track

        else:

            if pn.op in FXOPCODE_PLACEHOLDER:  # terminal condition
                partial_match = OrderedDict([(pn, dn)])
                matches.append(partial_match)

            elif pn.op in FXOPCODE_OUTPUT:

                if len(dn.all_input_nodes) == 0:  # it is impossible to continue the matching along the data graph: back-track
                    pass

                else:
                    partial_match = OrderedDict([(pn, dn)])
                    pn_ = next(iter(pn.all_input_nodes))
                    partial_matches = []
                    for dn_ in dn.all_input_nodes:
                        partial_matches += PathGraphMatcher._match_nodes(pattern, pn_, dn_, pattern_gm, data_gm)

                    if len(partial_matches) > 0:  # assemble partial candidate solution (PCS)
                        for pm in partial_matches:
                            map_ = partial_match.copy()
                            map_.update(**pm)
                            matches.append(map_)
                    else:  # no matches were found: back-track
                        pass

            else:  # we are matching the "core" nodes of the pattern

                if not PathGraphMatcher.is_body_node(dn) and not (pn in pattern.leakable_nodes):  # this node can't be modified, since it might be used in scopes other than the one defined by the pattern: back-track
                    pass

                else:
                    partial_match = OrderedDict([(pn, dn)])
                    partial_matches = PathGraphMatcher._match_nodes(pattern, next(iter(pn.all_input_nodes)), next(iter(dn.all_input_nodes)), pattern_gm, data_gm)

                    if len(partial_matches) > 0:  # assemble partial candidate solution (PCS)
                        for pm in partial_matches:
                            map_ = partial_match.copy()
                            map_.update(**pm)
                            matches.append(map_)
                    else:  # no matches were found: back-track
                        pass

        return matches

    def find(self, data_gm: fx.GraphModule) -> List[Match]:
        """Find the sub-graphs of the data graph matching the pattern graph.

        Note that this function makes the following assumption: that the IR of
        the data graph (i.e., its ``fx.Graph``) has been obtained using the
        same tracing function used to generate the query graph (i.e., the
        ``symbolic_trace_fn`` passed to this object's constructor method).

        From an algorithmic viewpoint, this function identifies matches using
        back-tracking.

        """

        # TODO: ensure that `data_gm == symbolic_trace_fn(root=data_gm)`,
        #       i.e., that the data graph has been obtained using the same
        #       tracing function as the query/pattern/template graph.

        matched_nodes = set()

        def overlaps_with_previous_matches(match: Match) -> bool:
            body_nodes = set(dn for pn, dn in match.nodes_map.items() if (pn.op not in FXOPCODES_IO))
            return any(n in matched_nodes for n in body_nodes)

        matches: List[Match] = []
        for data_anchor in reversed(data_gm.graph.nodes):

            candidate_matches = PathGraphMatcher._match_nodes(self.pattern, self._pattern_anchor, data_anchor, self.pattern.pattern_gm, data_gm)
            for candidate_match in candidate_matches:
                match = Match(anchor=data_anchor, nodes_map=candidate_match)
                if not overlaps_with_previous_matches(match):
                    matches.append(match)
                    body_nodes = set(dn for pn, dn in match.nodes_map.items() if pn.op not in FXOPCODES_IO)
                    matched_nodes = matched_nodes.union(body_nodes)
                else:
                    warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + "two matches with non-zero overlap were found; the latest will be discarded.")

        return matches
