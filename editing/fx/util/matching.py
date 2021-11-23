import copy

from torch import fx
from torch.fx.subgraph_rewriter import Match

__all__ = ['NonUniqueGeneralMatcher', 'SequentialMatcher', 'get_ordered_active_nodes']

class NonUniqueGeneralMatcher:
    def __init__(self, pattern : callable, trace : callable = fx.symbolic_trace):
        # Trace a GraphModule from pattern
        self.trace = trace
        p = self.trace(pattern)
        # we need to have access to both the pattern graph (using the output
        # node as an entry point) as well as the
        # enclosing GraphModule to correctly match call_module ops
        self.p = p
        self.pattern_anchor = next(iter(reversed(self.p.graph.nodes)))
        # this will be the graph module that we search for the pattern
        self.searched_gm : fx.GraphModule = None
        
    @property
    def searched_modules(self):
        # a dictionary of the modules contained in the searched GraphModule
        return dict(self.searched_gm.named_modules())

    @property
    def pattern_modules(self):
        # a dictionary of the modules contained in the pattern
        return dict(self.p.named_modules())

    def matches_subgraph_from_anchor(self, anchor : fx.Node):
        # similar to the fx method, except the nodes_map is not a member of the
        # class
        #TODO: is this really a nice way to return the results? (None if no
        # match, Match object if match)

        match = self._match_nodes(self.pattern_anchor, anchor)
        
        if match is None:
            return []
        try:
            mm = [Match(anchor=anchor, nodes_map=match)]
        except TypeError:
            mm = []
        return mm


    def _match_nodes(self, pn : fx.Node, gn : fx.Node):
        mapList = []
        mismatch = False
        match = self.__match_nodes(pn, gn)
        #import IPython; IPython.embed()
        # Eliminate malformed matches
        if match is not None:
            for key in list(pn.graph.nodes):
                if not key in match:
                    mismatch = True
            if not mismatch:
                pass
            else:
                return None
        else:
            return None    
        
        # Resolve placeholder
        copymatch = copy.deepcopy(match)
        for key, value in match.items():
            if isinstance(value, list):
                matched = False
                for listitem in value:
                    if listitem not in copymatch.values():
                        matched = True
                        match[key] = listitem
                        break
                if not matched:
                    return None
                
        return match
    
    # pn is the output node of the pattern graph
    # gn is the current lastmost node of the graph-to-be-matched
    def __match_nodes(self, pn : fx.Node, gn : fx.Node):
        # the following submethod is a modified version of the one from the
        # original SubgraphMatcher
        nodes_map = {}
        def attributes_are_equal(pn: fx.Node, gn: fx.Node) -> bool:
            # Use placeholder and output nodes as wildcards. The
            # only exception is that an output node can't match
            # a placeholder
            if (pn.op == "placeholder" or (pn.op == "output" and gn.op != "placeholder")):
                return True
            if (pn.op == 'get_attr' or pn.op == 'getattr'):
                return gn.op == 'get_attr' or gn.op == 'getattr' and pn.target == gn.target
            elif pn.op != "call_module":
                return pn.op == gn.op and pn.target == gn.target
            else:
                #print(gn)
                # for call_module op, we check that the called module's exact
                # type is the same - checking for the same target would require
                # their names to be the same which really makes no sense.
                return pn.op == gn.op and isinstance(self.searched_modules[gn.target], type(self.pattern_modules[pn.target]))
        if pn == ():
            return {}
#         elif pn.op == "placeholder":
#             nodes_map[pn] = gn
        elif not attributes_are_equal(pn, gn):
            return None
        else:
            nodes_map[pn] = gn
            for pattern_arg in pn._input_nodes.keys():
                matched = 0
                if pattern_arg.op != 'placeholder':
                    for graph_arg in gn._input_nodes.keys():
                        match = self.__match_nodes(pattern_arg, graph_arg)
                        if match is not None and graph_arg not in nodes_map.values():
                            # Merge dicts
                            matched = matched+1
                            nodes_map = {**nodes_map, **match}
                            continue
                else:
                    nodes_map[pattern_arg] = list(gn._input_nodes.keys())
                    
        return nodes_map
        
                        
    def match_graph(self, gm : fx.GraphModule):
        # this function returns a list of non-overlapping matches of self.p
        # in gm, which is first traced with self.trace. Any matches which
        # overlap previous matches are discarded.

        self.searched_gm = gm
        all_matches = []
        matched_nodes = set()
        def match_overlaps_with_previous(match):
            return any(n in matched_nodes and n.op not in ("placeholder", "output") and k.op not in ("placeholder", "output") for k, n in match.nodes_map.items())

        for node in self.searched_gm.graph.nodes:
            matches = self.matches_subgraph_from_anchor(node)
            #print("matches:  ", matches)
            for m in matches:
                if not match_overlaps_with_previous(m):
                    all_matches.append(m)
                    for k, n in m.nodes_map.items():
                        if k.op not in ("placeholder", "output"):
                            matched_nodes.add(n)
        return all_matches


class SequentialMatcher:
    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, pattern : callable, trace : callable = fx.symbolic_trace):
        # Trace a GraphModule from pattern
        self.trace = trace
        p = self.trace(pattern)
        # as this is a sequential matcher, ensure every node only has max. 1
        # input and output
        for n in p.graph.nodes:
            assert (n.op == 'placeholder' and len(n.users) == 1) or (n.op == 'output' and len(n.all_input_nodes) == 1) or (len(n.all_input_nodes) == 1 and len(n.users) == 1), "Only sequential patterns are supported!"
        # we need to have access to both the pattern graph (using the output
        # node as an entry point) as well as the
        # enclosing GraphModule to correctly match call_module ops
        self.p = p
        self.pattern_anchor = next(iter(reversed(self.p.graph.nodes)))
        # this will be the graph module that we search for the pattern
        self.searched_gm : fx.GraphModule = None

    @property
    def searched_modules(self):
        # a dictionary of the modules contained in the searched GraphModule
        return dict(self.searched_gm.named_modules())

    @property
    def pattern_modules(self):
        # a dictionary of the modules contained in the pattern
        return dict(self.p.named_modules())

    def matches_subgraph_from_anchor(self, anchor : fx.Node):
        # similar to the fx method, except the nodes_map is not a member of the
        # class
        #TODO: is this really a nice way to return the results? (None if no
        # match, Match object if match)
        matches = self._match_nodes(self.pattern_anchor, anchor)
        try:
            mm = [Match(anchor=anchor, nodes_map=m) for m in matches]
        except TypeError:
            mm = []
        return mm


    def _match_nodes(self, pn : fx.Node, gn : fx.Node, last_active_node : bool = False, nodes_map : dict = None):
        nodes_map = {} if nodes_map is None else nodes_map
        # as we do sequential traversal, the first step (checking if nodes
        # already traversed) of the original _match_nodes function can be
        # discarded

        # the following submethod is a modified version of the one from the
        # original SubgraphMatcher
        def attributes_are_equal(pn: fx.Node, gn: fx.Node) -> bool:
            # Use placeholder and output nodes as wildcards. The
            # only exception is that an output node can't match
            # a placeholder
            if (pn.op == "placeholder"
                    or (pn.op == "output" and gn.op != "placeholder")):
                return True
            if pn.op != "call_module":
                return pn.op == gn.op and pn.target == gn.target
            else:
                # for call_module op, we check that the called module's exact
                # type is the same - checking for the same target would require
                # their names to be the same which really makes no sense.
                return pn.op == gn.op and isinstance(self.searched_modules[gn.target], type(self.pattern_modules[pn.target]))

        # from here on, proceed as in the original implementation.
        if not attributes_are_equal(pn, gn):
            return None

        nodes_map[pn] = gn
        pn_users = [u for u in pn.users]

        if pn.op == "placeholder":
            return nodes_map
        # if we are in the "active" pattern, the graph node has to be
        # single-input and single-use
        # if (pn.op not in ("output", "placeholder") and
        # (len(gn.all_input_nodes) != 1) or (len(gn.users) > 1 and not
        # first_active_node)):
        if pn.op != "output" and ((len(gn.users) > 1 and not last_active_node) or len(gn.all_input_nodes) > 1):
            # if the gn has >1 users, the pattern is "leaking" and we don't
            # want to match it
            return None
        if pn.op == "output":
            # if the pattern node is an output, any of the branches leading to
            # the current graph node could be a potential match. we consider
            # all of them and return all matches.
            nodes_maps = []
            for gi in gn.all_input_nodes:
                nm = self._match_nodes(pn.all_input_nodes[0], gi, True, copy.copy(nodes_map))
                if nm is not None:
                    nodes_maps.append(nm)
            return nodes_maps
        # otherwise we are on a "matching track", so move one node down in
        # pattern and graph. We know that gn has only 1 input!
        return self._match_nodes(pn.all_input_nodes[0], gn.all_input_nodes[0], False, nodes_map)

    def match_graph(self, gm : fx.GraphModule):
        # this function returns a list of non-overlapping matches of self.p
        # in gm, which is first traced with self.trace. Any matches which
        # overlap previous matches are discarded.

        self.searched_gm = gm
        all_matches = []
        matched_nodes = set()
        def match_overlaps_with_previous(match):
            return any(n in matched_nodes and n.op not in ("placeholder", "output") and k.op not in ("placeholder", "output") for k, n in match.nodes_map.items())

        for node in self.searched_gm.graph.nodes:
            #print(node)
            matches = self.matches_subgraph_from_anchor(node)
            #print("matches:  ", matches)
            for m in matches:
                #print("m:  ", m)
                if not match_overlaps_with_previous(m):
                    all_matches.append(m)
                    for k, n in m.nodes_map.items():
                        if k.op not in ("placeholder", "output"):
                            matched_nodes.add(n)
        return all_matches

def get_ordered_active_nodes(m : Match):
    return [v for v in m.nodes_map.values()][-2:0:-1]
