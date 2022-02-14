import copy
import torch.nn as nn
import torch.fx as fx
from torch.fx.subgraph_rewriter import Match, SubgraphMatcher
from typing import List, Dict, Union, Callable

from quantlib.newutils import quantlib_err_header


_OPCODE_PLACEHOLDER = {'placeholder'}
_OPCODE_OUTPUT      = {'output'}
_OPCODE_CALL_MODULE = {'call_module'}


class LinearGraphMatcher(SubgraphMatcher):

    def __init__(self,
                 symbolic_trace_fn: Callable,
                 pattern: nn.Sequential):
        """An object to identify linear sub-graphs into a target graph.

        **Linear graphs** are amongst the simplest directed graphs. A linear
        graph is a graph :math:`G = (V_{G}, E_{G})` such that there exists an
        enumeration :math:`(v_{0}, \dots, v_{|V_{G}| - 1})` of :math:`V_{G}`
        according to which :math:`E_{G} = \{ (v_{i}, v_{i+1}), i = 0, \dots,
        |V_{G}| - 2 \}`. In other words, a linear graph is an acyclic graph
        such that:

        * there is a single node :math:`v_{0}` which has exactly zero inbound
          nodes and exactly one outbound node (the *entry point*);
        * there is a single node :math:`v_{|V_{G}| - 1}` which has exactly one
          inbound node and exactly zero outbound nodes (the *exit point*);
        * all the other nodes have exactly one inbound and one outbound node.

        Despite their simplicity, these graphs represent a wide range of
        sub-graphs used in computational graphs. In PyTorch, linear graphs are
        described using the ``nn.Sequential`` construct, which executes the
        ``nn.Module``s passed to its constructor one after the other.

        **Singleton graphs** are a special sub-case of linear graphs. Indeed,
        singleton graphs contain a single vertex :math:`v_{0}` and have no
        arcs, i.e., :math:`E_{G} = \emptyset`.

        """

        self._symbolic_trace_fn = symbolic_trace_fn

        # trace the pattern `nn.Module`
        self._pattern_gm: fx.GraphModule = self._symbolic_trace_fn(root=pattern)
        if not LinearGraphMatcher.is_path_graph(self._pattern_gm.graph):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "requires a path graph as template.")
        super(LinearGraphMatcher, self).__init__(pattern=self._pattern_gm.graph)  # this instruction has the side-effect of pinning `pattern_graph` to the `self.pattern` attribute

        # decompose the nodes of the template part (L-term) of the graph rewriting rule
        self._Kterm_entry = tuple(pn for pn in self.pattern.nodes if pn.op in _OPCODE_PLACEHOLDER)
        self._Kterm_exit  = tuple(pn for pn in self.pattern.nodes if pn.op in _OPCODE_OUTPUT)
        self._Lterm_core  = tuple(pn for pn in self.pattern.nodes if (pn not in self._Kterm_entry) and (pn not in self._Kterm_exit))

        # these parameters are used when the `LinearGraphMatcher` is searching for application points
        self._is_searching: bool = False
        self._target_gm: Union[None, fx.GraphModule] = None

    @property
    def pattern_gm(self) -> fx.GraphModule:
        return self._pattern_gm

    @property
    def target_gm(self) -> fx.GraphModule:
        if not self._is_searching:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "should not access _target_gm when it is not searching for application points.")
        return self._target_gm

    @staticmethod
    def is_path_graph(g: fx.Graph) -> bool:

        # define helper functions

        def _check_placeholder(n: fx.Node) -> bool:
            assert len(n.all_input_nodes) == 0
            return len(n.users) == 1  # has exactly one output

        def _check_output(n: fx.Node) -> bool:
            assert len(n.users) == 0
            return len(n.all_input_nodes) == 1  # has exactly one input

        def _check_others(n: fx.Node) -> bool:
            cond_in  = len(n.all_input_nodes) == 1  # has exactly one input
            cond_out = len(n.users) == 1            # has exactly one output
            return cond_in & cond_out

        # main check loop

        state = True

        for node in g.nodes:

            if node.op in _OPCODE_PLACEHOLDER:
                state &= _check_placeholder(node)
            elif node.op in _OPCODE_OUTPUT:
                state &= _check_output(node)
            else:
                state &= _check_others(node)

            if state is False:
                break

        return state

    # Compare the pattern node `pn` against the target node `tn`
    def _match_nodes(self, pn: fx.Node, tn: fx.Node) -> bool:

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in self.nodes_map:
            return self.nodes_map[pn] == tn

        def attributes_are_equal(pn: fx.Node, tn: fx.Node) -> bool:

            # Use placeholder and output nodes as wildcards. The
            # only exception is that an output node can't match
            # a placeholder
            if pn.op in ('placeholder',):
                state = True
            elif pn.op in ('output',):
                if tn.op not in ('placeholder',):
                    state = True
                else:
                    state = False

            elif pn.op in ('call_module',):
                # The `torch.fx` implementation of `SubgraphMatcher` check
                # that the `target` attributes of candidate matching nodes are
                # the same. We are instead interested in determining whether
                # the two nodes represent instances of the same class.
                cond_op   = pn.op == tn.op
                if cond_op:
                    pm_class  = type(dict(self.pattern_gm.named_modules())[pn.target])
                    tm_object = dict(self.target_gm.named_modules())[tn.target]
                    cond_type = isinstance(tm_object, pm_class)
                else:
                    cond_type = False
                state = cond_op & cond_type

            else:
                cond_op     = pn.op == tn.op
                cond_target = pn.target == tn.target
                state = cond_op & cond_target

            return state

        # Terminate early if the node attributes are not equal
        if not attributes_are_equal(pn, tn):
            match_found = False
            return match_found

        # Optimistically mark `pn` as a match for `tn`
        self.nodes_map[pn] = tn

        # Traverse the use-def relationships to ensure that `pn` is a true
        # match for `tn`
        if pn.op in ('placeholder',):
            match_found = True

        elif pn.op in ('output',):
            match_found = any(self._match_nodes(pn.all_input_nodes[0], tn_) for tn_ in tn.all_input_nodes)

        elif len(pn.all_input_nodes) != len(tn.all_input_nodes):
            match_found = False
            return match_found

        else:
            cond_n_inputs = len(pn.all_input_nodes) == len(tn.all_input_nodes)
            cond_all_matches = all(self._match_nodes(pn_, tn_) for pn_, tn_ in zip(pn.all_input_nodes, tn.all_input_nodes))
            match_found = cond_n_inputs and cond_all_matches

        if not match_found:
            self.nodes_map.pop(pn)

        return match_found

    def find_application_points(self, target_gm: fx.GraphModule) -> List[Match]:

        # ensure that the IR is the same as the one representing the pattern graph
        self._target_gm = self._symbolic_trace_fn(root=target_gm)
        self._is_searching = True

        matches: List[Match] = []
        matched_nodes = set()

        def overlaps_with_previous_matches(match: Match) -> bool:
            core_nodes = tuple(n for k, n in match.nodes_map.items() if (k.op not in ('placeholder', 'output')) and (n.op not in ('placeholder', 'output')))
            return any(n in matched_nodes for n in core_nodes)

        # Consider each node as an "anchor" (deepest matching graph node)
        for anchor in self._target_gm.graph.nodes:

            if self.matches_subgraph_from_anchor(anchor):

                def pattern_is_contained(nodes_map: Dict[fx.Node, fx.Node]) -> bool:
                    # `lookup` represents all the nodes in `original_graph`
                    # that are part of `pattern`
                    lookup: Dict[fx.Node, fx.Node] = {v: k for k, v in nodes_map.items()}
                    for n in lookup.keys():

                        # Nodes that can "leak"...

                        # Placeholders (by definition)
                        if n.op == "placeholder":
                            continue
                        # Pattern output (acts as a container)
                        if lookup[n].op == "output":
                            continue
                        # Result contained by pattern output (what we'll
                        # hook in to the new Graph, thus what we'll
                        # potentially use in other areas of the Graph as
                        # an input Node)
                        if (len(lookup[n].users) == 1) and (list(lookup[n].users.keys())[0].op == "output"):  # TODO: why not using `next(iter([...].keys()))` as opposed to `list([...].keys())[0]`?
                            continue

                        for user in n.users:
                            # If this node has users that were not in
                            # `lookup`, then it must leak out of the
                            # pattern subgraph
                            if user not in lookup:
                                return False

                    return True

                if pattern_is_contained(self.nodes_map):

                    candidate_match = Match(anchor=anchor, nodes_map=copy.copy(self.nodes_map))
                    if not overlaps_with_previous_matches(candidate_match):
                        matches.append(candidate_match)
                        matched_nodes.union(set(tn for pn, tn in candidate_match.nodes_map.items() if pn not in ('placeholder', 'output')))
                    else:
                        print('Warning: match already found!')

        self._is_searching = False
        self._target_gm = None

        return matches
