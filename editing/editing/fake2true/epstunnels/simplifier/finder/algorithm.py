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

import torch
import torch.fx as fx
from typing import Set, List

from .whitelists import whitelist_call_module, whitelist_call_method, whitelist_call_function
from ..applicationpoint import CandidateEpsTunnelConstruct
from quantlib.editing.graphs.fx import FXOpcodeClasses
from quantlib.editing.graphs.nn import EpsTunnel


# -- TOPOLOGICAL CHECK -- #

def check_node(n: fx.Node, gm: fx.GraphModule) -> bool:
    """Verify whether an ``fx.Node`` does not modify the scales of its inputs.

    This function uses a whitelisting logic: in order to pass the check, the
    argument ``fx.Node`` must be explicitly defined in some whitelist. We
    define a whitelist for each ``torch.fx`` opcode.

    Whitelists are implemented as dictionaries. To pass the whitelisting
    check, an ``fx.Node``'s type must appear amongst the keys of the
    corresponding dictionary. Users can define more restrictive conditions by
    implementing lists of checker functions; a checker function should take an
    ``fx.Node`` as input and return a Boolean.

    """

    # we condition the checks on the `fx.Node`'s opcode
    opcode = n.op

    if opcode in FXOpcodeClasses.IO.value:
        state = False

    elif opcode in FXOpcodeClasses.CALL_MODULE.value:
        m = gm.get_submodule(target=n.target)
        if isinstance(m, tuple(whitelist_call_module.keys())):
            state = True if all(c(n) for c in whitelist_call_module[type(m)]) else False
        else:
            state = False

    elif opcode in FXOpcodeClasses.CALL_METHOD.value:
        if n.target in whitelist_call_method.keys():
            state = True if all(c(n) for c in whitelist_call_method[n.target]) else False
        else:
            state = False

    elif opcode in FXOpcodeClasses.CALL_FUNCTION.value:
        t = n.target.__name__
        if t in whitelist_call_function.keys():
            state = True if all(c(n) for c in whitelist_call_function[t]) else False
        else:
            state = False

    else:
        state = False

    return state


def find_backward_frontier(n: fx.Node, gm: fx.GraphModule) -> Set[fx.Node]:

    # impacted `EpsTunnel`s (ancestors)
    B = set(filter(lambda p: (p.op in FXOpcodeClasses.CALL_MODULE.value) and isinstance(gm.get_submodule(target=p.target), EpsTunnel), n.all_input_nodes))

    early_exit = False

    other_predecessors = set(n.all_input_nodes).difference(B)
    for p_ in other_predecessors:  # scan non-`EpsTunnel` predecessors

        if check_node(p_, gm):
            P = find_backward_frontier(p_, gm)
            if len(P) == 0:  # the traversal up this ancestor "leaked"
                early_exit = True
            else:
                B = B.union(P)
        else:
            early_exit = True

        if early_exit:
            B = set()  # notify caller that the traversal path "leaked"
            break

    return B


def find_forward_frontier(n: fx.Node, gm: fx.GraphModule) -> Set[fx.Node]:

    # impacted `EpsTunnel`s (descendants)
    F = set(filter(lambda s: (s.op in FXOpcodeClasses.CALL_MODULE.value) and isinstance(gm.get_submodule(target=s.target), EpsTunnel), n.users))

    early_exit = False

    other_successors = set(n.users).difference(F)
    for s_ in other_successors:  # scan non-`EpsTunnel` successors

        if check_node(s_, gm):
            S = find_forward_frontier(s_, gm)
            if len(S) == 0:  # the traversal down this descendant "leaked"
                early_exit = True
            else:
                F = F.union(S)
        else:
            early_exit = True

        if early_exit:
            F = set()  # notify caller that the traversal path "leaked"
            break

    return F


def find_candidate_construct_from_anchor(anchor: fx.Node, gm: fx.GraphModule) -> CandidateEpsTunnelConstruct:
    """Find a candidate``EpsTunnel`` construct.

    Given an ``fx.Node`` representing an ``EpsTunnel``, this function
    identifies the generating frontier of its enclosing ``EpsTunnel``
    construct.

    This function uses a backward-forward graph traversal until it reaches a
    fixed-point.
      * The initial data of each iteration is a portion of the candidate
        ``EpsTunnel`` construct's outbound frontier, the *tentative outbound
        frontier*.
      * For each node in this portion, the algorithm traverses the data graph
        backward until it impacts on ``EpsTunnel``s, or some path traverses an
        ``fx.Node`` for which the preservation of the scale is not guaranteed.
        In the first case, the collection of discovered ``EpsTunnel``s is the
        tentative inbound frontier.
      * For each node in the tentative inbound frontier, the algorithm
        traverses the data graph forward until it impacts on ``EpsTunnel``s,
        or some path traverses an ``fx.Node`` for which the preservation of
        the scale is not guaranteed. In the first case, the collection of
        discovered ``EpsTunnel``s is the new tentative outbound frontier.
      * If at any moment a traversal visits an ``fx.Node`` for which the
        preservation of the scale is not guaranteed, the tentative frontiers
        are flushed, and empty sets signal that the search for an
        ``EpsTunnel`` construct has failed.
      * If the algorithm reaches a fixed-point (i.e., the tentative inbound
        and outbound frontiers do not change between two consecutive
        iterations), the algorithm has identified a candidate construct.

    """

    # tentative frontiers
    B = set()
    F = {anchor}

    early_exit = False

    while True:  # emulate a `do-while` construct

        B_old, F_old = B, F

        # compute ancestor sub-graph (backward pass -- from outbound tentative frontier)
        B_subsets = list(map(lambda n: find_backward_frontier(n, gm), F_old))
        if any(len(b) == 0 for b in B_subsets):  # a traversal path "leaked" out of the construct
            early_exit = True
        else:
            B = set().union(*B_subsets)

        # compute descendant sub-graph (forward pass -- from inbound tentative frontier)
        F_subsets = list(map(lambda n: find_forward_frontier(n, gm), B))
        if any(len(f) == 0 for f in F_subsets):  # a traversal path "leaked" out of the construct
            early_exit = True
        else:
            F = set().union(*F_subsets)

        # does the candidate construct "leak"?
        if early_exit:
            B = set()
            F = set()
            break

        if (B == B_old) and (F == F_old):  # fixed-point found!
            break

    return CandidateEpsTunnelConstruct(backward=B, forward=F)


def find_candidate_constructs(gm: fx.GraphModule) -> List[CandidateEpsTunnelConstruct]:

    constructs = []

    V_eps = list(filter(lambda n: (n.op in FXOpcodeClasses.CALL_MODULE.value) and (isinstance(gm.get_submodule(target=n.target), EpsTunnel)), list(reversed(gm.graph.nodes))))
    V_visited = set()  # we avoid visiting the same `EpsTunnel` twice in its role of outbound frontier member

    while len(V_eps) > 0:

        anchor = V_eps.pop(0)  # https://docs.python.org/3/tutorial/datastructures.html
        construct = find_candidate_construct_from_anchor(anchor, gm)

        if construct.is_empty():
            V_visited.add(anchor)

        else:
            assert anchor in construct.forward
            if len(construct.forward.intersection(set(V_visited))) > 0:  # an `EpsTunnel` can be part of at most one inbound and outbound frontier
                raise RuntimeError
            else:
                V_eps = [n for n in V_eps if n not in construct.forward]  # we use list comprehension to filter visited nodes to preserve topological ordering
                constructs.append(construct)

    return constructs


# -- SEMANTIC CHECK -- #

def verify_candidate_construct(cc: CandidateEpsTunnelConstruct, gm: fx.GraphModule) -> bool:
    """Verify the semantic of a candidate ``EpsTunnel`` construct.

    This function checks that a topologically valid ``EpsTunnel`` construct
    can be semantically simplified.

    """

    input_scales  = tuple(map(lambda n: gm.get_submodule(target=n.target).eps_out, cc.backward))
    output_scales = tuple(map(lambda n: gm.get_submodule(target=n.target).eps_in,  cc.forward))

    all_scales = (*input_scales, *output_scales)

    scale_pairs = zip(all_scales[:-1], all_scales[1:])  # since equality is transitive, it suffices to make pair-wise comparison instead of checking the whole Cartesian product
    cond_shape  = all(s1.shape == s2.shape for s1, s2 in scale_pairs)
    cond_values = all(torch.all(s1 == s2)  for s1, s2 in scale_pairs)

    return cond_shape and cond_values
