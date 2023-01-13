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

"""Some rewriting rules look for classes of patterns that it is unfeasible or
simply not convenient to fully enumerate. In these cases, we replace pattern
matching with specific algorithms that can capture any pattern that we are
interested into.

An example of such patterns is that of variadic operations (i.e., operations
accepting different numbers of inputs). Examples of such operations include
reduction operators, such as addition. Due to commutativity, the order in
which we pass addends does not matter; tt is sufficent to know which numbers
should be added. In fact, we can put them on a stack, randomly permute the
stack, then pop the first two and push the result back to the stack: it does
not matter in which order we permute the stack, the result will always be the
same. These kind of permutation-independent sequences of binary operations can
be abstracted away as operations acting on stacks of operands of arbitrary
size. This is usually how variadic operations are created.

Continuing with the addition example but going back to the pattern matching
context, to pursue a pattern-based matching of variadic attitions we would
need to enumerate all possible depth-one trees, but this is unfeasible since
the number :math:`n` of children can grow to :math:`\infty`!

In these cases, finding the sub-graphs that we are interested in must rely on
invariants that are easier to detect. In the case of additions, we might want
to search for all sub-graphs:
* that are trees;
* whose root node represents an addition.

Since this construction can be generalised further than additions, this module
provides two abstractions:
* ``OpTree``s, to describe trees of operations rooted at a given ``fx.Node``;
* ``OpSpec``s, to express all the possible declinations of a given abstract
  operation; an ``OpSpec`` maps ``torch.fx`` opcodes to the corresponding
  targets (where *target* is meant in the ``fx.Node`` sense).

"""

from __future__ import annotations

from collections import OrderedDict
import torch
import torch.fx as fx
from typing import Tuple, List, Any, Union, Callable

from ..base import ApplicationPoint


class OpTree(ApplicationPoint):

    def __init__(self, root: fx.Node):

        # start from the node which is most-donwstream from the point-of-view of the computational graph
        self._root:  fx.Node = root
        self._nodes: List[fx.Node] = [self._root]

    @property
    def root(self) -> fx.Node:
        return self._root

    @property
    def nodes(self) -> List[fx.Node]:
        return self._nodes

    def merge(self, other_optrees: Union[OpTree, List[OpTree]]) -> None:
        """This function operates by side-effect on ``self._nodes``."""
        # validate input type
        if not (isinstance(other_optrees, OpTree) or (isinstance(other_optrees, list) and all(isinstance(item_, OpTree) for item_ in other_optrees))):
            raise ValueError

        # canonicalise input
        if isinstance(other_optrees, OpTree):
            other_optrees = [other_optrees]

        for optree in other_optrees:
            self._nodes.extend(optree.nodes)

    @property
    def inbound_frontier(self) -> Tuple[fx.Node, ...]:
        """Compute the ``fx.Node``s that are inputs to the ``OpTree``.

        Note that there the order of argument traversal depends on the order
        of the ``fx.Node``s stored in ``self._nodes``. This base class does
        not enforce any ordering on the insertion of ``fx.Node``s into
        ``self._nodes``, therefore there is no guarantee on the ordering of
        ``inbound_frontier`` items: functions using this attribute should not
        make any assumption on its ordering.

        """

        def is_fxnode_or_fxnode_container(arg: Any) -> bool:
            is_fxnode = isinstance(arg, fx.Node)
            is_fxnode_container = isinstance(arg, (tuple, list,)) and all(isinstance(item_, fx.Node) for item_ in arg)
            return is_fxnode or is_fxnode_container

        inbound_frontier = [
                               arg for node in self._nodes
                               for arg in node.args if (arg not in self._nodes) and is_fxnode_or_fxnode_container(arg)
                           ] + [
                               v for node in self._nodes
                               for v in node.kwargs.values() if (v not in self._nodes) and is_fxnode_or_fxnode_container(v)
                           ]

        # I want a flat data structure, i.e., no item in the output iterable
        # should be a container of `fx.Node`s (e.g., `torch.concat` calls take
        # as inputs iterables of `torch.Tensor`s).
        inbound_frontier_flattened: List[fx.Node] = []
        for arg in inbound_frontier:
            if isinstance(arg, (tuple, list,)):
                inbound_frontier_flattened.extend([a for a in arg])
            else:
                inbound_frontier_flattened.append(arg)

        return tuple(inbound_frontier_flattened)


FXNodeTargetType = Union[str, Callable[[Tuple[torch.Tensor, ...]], torch.Tensor]]


class OpSpec(OrderedDict):
    """``OpSpec`` is meant to be inherited from, and children classes must
    specify at least one valid key-value pair.
    """

    def __setitem__(self, opcode: str, targets: Tuple[FXNodeTargetType, ...]):

        # validate types
        if not isinstance(opcode, str):
            raise TypeError
        if not (isinstance(targets, str) or callable(targets) or (isinstance(targets, tuple) and all((isinstance(item_, str) or callable(item_)) for item_ in targets))):
            raise ValueError

        # canonicalise `targets` argument
        if isinstance(targets, str) or callable(targets):
            targets = (targets,)

        super(OpSpec, self).__setitem__(opcode, targets)

    def matches_opspec(self, dn: fx.Node) -> bool:
        return (dn.op in self.keys()) and (dn.target in self[dn.op])
