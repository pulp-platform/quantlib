"""
A fake-quantised (FQ) array is associated to two quantities:

* its underlying **integer range** :math:`Q`, which is specified by an integer
  **precision** :math:`B > 0` and an integer **offset** :math:`z \in
  \mathbb{Z}`;
* its **representation quantum**, a positive floating-point number
  :math:`\epsilon`.

We therefore express an FQ array as :math:`x = \hat{x} \epsilon_{x}`, where
:math:`\epsilon_{x}` is the quantum and :math:`\hat{x}_{i} \in Q_{x}`, with
:math:`Q_{x}` being the integer range associated with the array.

We will now discuss two example binary operations with FQ quantised arrays
that show why and how (quantitavely speaking) the quanta associated with the
two arrays impact the performance of the corresponding operations. For
simplicity, we will consider to apply the operations to two one-dimensional
FQ arrays :math:`a, b` with :math:`N > 0` components each.

The first example is the element-wise sum between :math:`a` and :math:`b`.
When both arrays share the same quantum, i.e., when :math:`\epsilon_{a} =
\epsilon_{b} = \epsilon_{c}`, we can rewrite

.. math::

   \begin{split}
   c_{i}
   &= a_{i} + b_{i} \\
   &= \hat{a}_{i} \epsilon_{a} + \hat{b}_{i} \epsilon_{b} \\
   &= \epsilon_{c} (\hat{a}_{i} + \hat{b}_{i})
   \end{split}

by means of the distributive property of real multiplication. From this
equation we notice the following:
* when the arrays do not share the same quantum, adding them up requires
  :math:`N` floating-point additions;
* with respect to the original :math:`N` floating-point additions, the
  reformulation requires :math:`N` integer additions;
* identifying the integer ranges of :math:`a` and :math:`b` respectively with
  the symbols :math:`Q_{a}, Q_{b}` and assuming that they both have zero
  offsets, the integer range of :math:`c` will also have zero offset, whereas
  its precision will be :math:`B_{c} = \max\{ B_{a}, B_{b} \} + 1`.

The second example is the dot-product between :math:`a` and :math:`b`.
Performing the operation between the fake-quantised arrays requires :math:`N`
floating-point multiplicatoins and :math:`N-1` floating-point additions (or
:math:`N` floating-point multiply-accumulate, MAC, operations). By using the
elementary commutative, associative, and distributive property (in this order)
of real multiplication, we can reformulate the dot product to use only two
floating-point multiplications, :math:`N` integer multiplications and
:math:`N-1` integer additions (or :math:`N` integer MAC operations):

.. math::

   \begin{split}
   \langle a, b \rangle
   &= \sum_{i = 1}^{N} a_{i} b_{i} \\
   &= \sum_{i = 1}^{N} \hat{a}_{i} \epsilon_{a} \hat{b}_{i} \epsilon_{b} \\
   &= \sum_{i = 1}^{N} \epsilon_{a} \epsilon_{b} \hat{a}_{i} \hat{b}_{i} \\
   &= \sum_{i = 1}^{N} (\epsilon_{a} \epsilon{b}) (\hat{a}_{i} \hat{b}_{i}) \\
   &= \epsilon_{a} \epsilon_{b} \sum_{i = 1}^{N} \hat{a}_{i} \hat{b}_{i} \,.
   \end{split}

Let now :math:`\epsilon_{A_{1}} \neq \epsilon_{A_{2}}` be two quanta, and let
:math:`A_{1}, A_{2}` be a partition of :math:`\{ 1, \dots, N \}` such that all
the components of :math:`a` with index in :math:`A_{1}` have quantum
:math:`\epsilon_{A_{1}}`, and all the components with index in :math:`A_{2}`
have quantum :math:`\epsilon_{A_{2}}`. In more formal terms and for an
arbitrary partition :math:`\{ A_{j} \}` of :math:`\{ 1, \dots, N \}`, if we
define

.. math::

   \begin{split}
   \chi_{A_{j}} \,:\,
   \{ 1, \dots, N \} &\to \{ 0, 1 \} \\
   i &\mapsto
   \begin{cases}
     0, \,\text{if } i \notin A_{j}, \\
     1, \,\text{if } i \in A_{j},
   \end{cases}
   \end{split}

to be the indicator function of the :math:`j`-th partition, we define the
**component quantum** to be the function

.. math::

   \epsilon_{\{ A_{j} \}}
   := \sum_{j = 1}^{| \{ A_{j} \} |} \chi_{A_{j}} \epsilon_{A_{j}} \,.

In this formalism, we can see that the dot product between :math:`a` and
:math:`b` has to be rewritten:

.. math::

   \begin{split}
   \langle a, b \rangle
   &= \sum_{i = 1}^{N} a_{i} b_{i} \\
   &= \sum_{i = 1}^{N} \hat{a}_{i} \epsilon_{\{ A_{1}, A_{2} \}}(i) \hat{b}_{i} \epsilon_{b} \\
   &= \sum_{i \in A_{1}} \hat{a}_{i} \epsilon_{A_{1}} \hat{b}_{i} \epsilon_{b} + \sum_{i \in A_{2}} \hat{a}_{i} \epsilon_{A_{2}} \hat{b}_{i} \epsilon_{b} \\
   &= \epsilon_{A_{1}} \epsilon_{b} \sum_{i \in A_{1}} \hat{a}_{i} \hat{b}_{i} + \epsilon_{A_{2}} \epsilon_{b} \sum_{i \in A_{2}} \hat{a}_{i} \hat{b}_{i} \,.
   \end{split}

From this equation, we see that computing the dot product requires :math:`N`
integer multiplications or :math:`N - 2` integer additions (or still :math:`N`
MACs in total), but four floating-point multiplications (although they can be
reduced to three by collecting the multiplications by :math:`\epsilon_{b}`.
For more general partitions on both :math:`a`'s and :math:`b`'s quanta, we
will generate :math:`| \{ A_{j} \} | | \{ B_{k} \} |` subsets of :math:`\{ 1,
\dots, N \}` (since each index can belong to exactly one item from each
partition), and each subset will require two floating-point multiplications.
This second discussion should have shown why it is important to ensure that as
many elements as possible inside an array share the same precision.

The first example (element-wise addition of two FQ arrays) highlights the
importance of equating the quanta of two FQ arrays before summing them up. The
second example (dot product between two FQ arrays) hightlights the necessity
of reducing the heterogeneity of the quanta associated with an array; in
practical contexts, these heterogeneous quanta inside the same array arise
when concatenating (or stacking) FQ arrays which have different quanta. For
these reasons, this module introduces abstractions that traverse computational
graphs of QNNs ensuring that element-wise additions between FQ arrays process
arrays which share the same quantum, and that concatenation operations produce
arrays with homogeneous quanta (which facilitate the integerisation of
downstream linear operations, i.e., operations involving dot products).
"""

from collections import namedtuple
import operator
import torch
# import torch.nn as nn
import torch.fx as fx
from typing import Optional, Union, Tuple, List

from quantlib.utils import quantlib_err_header


NodeSpec = namedtuple('NodeSpec', ['op', 'targets'])


class OpTree(object):

    def __init__(self, root: fx.Node):

        # start from the node which is most-donwstream from the point-of-view of the computational graph
        self._root = root
        self._nodes = [self._root]

        # root may get deleted afterwards, therefore this is the best moment to note down information about it
        self._root_args = self.inbound_frontier
        self._root_users = [u for u in self._root.users]  # root may get deleted: I need to note down NOW which other dependant nodes might get useless

    @property
    def root(self) -> fx.Node:
        return self._root

    @property
    def nodes(self) -> List[fx.Node]:
        return self._nodes

    @property
    def inbound_frontier(self) -> Tuple[fx.Node]:
        """Compute the `fx.Node`s that are input to the `OpTree`.

        The flattening assumes that the order of arguments and keyword
        arguments is not important (i.e., information about the order of the
        arguments is not retained from the output data structure).
        """

        def is_fxnode_or_fxnode_container(arg) -> bool:
            cond_fxnode = isinstance(arg, fx.Node)
            cond_fxnode_container = isinstance(arg, (tuple, list)) and all(isinstance(item_, fx.Node) for item_ in arg)
            return cond_fxnode or cond_fxnode_container

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

        inbound_frontier_flattened = []
        for arg in inbound_frontier:
            if isinstance(arg, (list, tuple)):
                inbound_frontier_flattened.extend([a for a in arg])
            else:
                inbound_frontier_flattened.append(arg)

        return tuple(inbound_frontier_flattened)

    def add_node(self, node: fx.Node):
        if node in self.nodes:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "should not be registering a node that has already been registered (possible non-DAG).")
        self.nodes.append(node)


class OpTreeMatcher(object):

    def __init__(self,
                 node_specs: List[NodeSpec]):
        super(OpTreeMatcher, self).__init__()
        self._node_specs = node_specs

    @staticmethod
    def _matches_spec(node: fx.Node, node_spec: NodeSpec):
        return (node.op == node_spec.op) and (node.target in node_spec.targets)

    @staticmethod
    def _matches_any_spec(node: fx.Node, node_specs: List[NodeSpec]):
        return any(map(lambda ns: OpTreeMatcher._matches_spec(node, ns), node_specs))

    @staticmethod
    def _find_op_trees(node:           fx.Node,
                       node_specs:     List[NodeSpec],
                       visited_nodes:  Optional[set],
                       current_optree: Optional[OpTree],
                       optrees:        List[OpTree]):
        """Detect all trees whose nodes are of specified type(s).

        This function will return a list of application points for the
        harmonisation GRR. Each application point is an object of class
        `OpTree`.

        No `fx.Node` can belong to more than one `OpTree` at a time. Thanks to
        a depth-first traversal logic, the `OpTree`s returned by the function
        satisfy the following property: that the root of the :math:`i`-th `OpTree` is
        topologically antecedent to the root of the :math:`(i+1)`-th `OpTree`.
        This property can be used during the application part of the GRR, when
        the quanta are harmonised proceeding downstream from the network's
        input.
        """

        if node not in visited_nodes:

            visited_nodes.add(node)

            if current_optree is not None:  # I am traversing an `OpTree`

                if len(node.users) == 0:
                    raise RuntimeError(quantlib_err_header(obj_name=OpTreeMatcher.__name__) + "should not be performing recursive calls on `fx.Node`s which are not inputs to other operations. Who triggered the recursive call?")

                if OpTreeMatcher._matches_any_spec(node, node_specs):  # I am still inside the `OpTree` from the previous stack frame

                    if len(node.users) == 1:  # continue the traversal of the same `OpTree`
                        optree = current_optree
                        optree.add_node(node)

                    else:  # `1 < len(node.users)`: I have found a branching point in the network topology, which I use as the root for a new `OpTree`
                        optree = OpTree(root=node)

                else:  # I have exited the current `OpTree`; I will continue traversal down this node looking for new roots
                    optree = None

            else:  # `current_optree is None`

                if OpTreeMatcher._matches_any_spec(node, node_specs):  # I am possibly entering a new tree
                    optree = OpTree(root=node)

                else:
                    optree = None

            # depth-first recursive call
            for input_ in node.all_input_nodes:
                OpTreeMatcher._find_op_trees(input_, node_specs, visited_nodes, optree, optrees)

            # only flush real `OpTree`s to output when the search down its root is completed
            if optree is not None:
                if node is optree.root:
                    optrees.append(optree)

        else:  # `node in visited_nodes`: this node has already been visited
            pass

        return

    def find_application_points(self, data_gm: fx.GraphModule) -> List[OpTree]:

        visited_nodes = set()
        optree: Union[OpTree, None] = None
        optrees: List[OpTree] = []

        for n in reversed(data_gm.graph.nodes):
            OpTreeMatcher._find_op_trees(n, self._node_specs, visited_nodes, optree, optrees)

        return optrees

    #     # apply the rewriting to all the application points
    #     for i, optree in enumerate(optrees):
    #
    #         # create the replacement graph...
    #         module = self.replacement_fn(optree)
    #
    #         # ...add it to the graph...
    #         new_target = f"_QL_OP_TREE_REPLACE_{self.name.upper() + '_' if self.name != '' else ''}{i}"
    #         gm.add_submodule(new_target, module)
    #         with gm.graph.inserting_before(optree.root):  # add a node for the submodule call
    #             new_node = gm.graph.call_module(new_target, args=optree.inbound_frontier)
    #         optree.root.replace_all_uses_with(
    #             new_node)  # attach the module to the previous users of the tree's end node
    #
    #         # ...and finally delete the "dead code"
    #         for node in optree.nodes:
    #             gm.graph.erase_node(node)
    #
    #     return gm


class AddTreeMatcher(OpTreeMatcher):

    def __init__(self):
        node_specs = [
            NodeSpec('call_function', (torch.add, operator.add)),
            NodeSpec('call_method',   ('add',)),
        ]
        super(AddTreeMatcher, self).__init__(node_specs=node_specs)


class ConcatTreeMatcher(OpTreeMatcher):

    def __init__(self):
        node_specs = [
            NodeSpec('call_function', (torch.cat,)),
            NodeSpec('call_function', (torch.stack,))
        ]
        super(ConcatTreeMatcher, self).__init__(node_specs=node_specs)
