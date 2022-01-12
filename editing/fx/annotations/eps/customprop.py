import collections
import torch
import torch.nn as nn
import torch.fx as fx

from typing import List, Tuple, Dict, Any, Union

import quantlib.algorithms  # TODO: establish a smarter, more concise way of importing all `nn.Module`s from a `quantlib.algorithm` namespace
import quantlib.editing.fx


def _unpack_fxnode_inputs(n: fx.Node) -> Tuple[List[Tuple[int, fx.Node]], List[Tuple[int, Any]], Dict[str, fx.Node], Dict[str, Any]]:
    """Retrieve and partition the inputs to an `fx.Node`.

    Each `fx.Node` represents the execution of a (symbolic) operation in the
    flow of the target network. Each operation is implemented as a Python
    function. The signature of a Python function can be though of as an
    immutable :math:`N`-tuple of the form

    .. math:
       ((1, m_{1}, t_{1}), \dots, (N, m_{N}, t_{N})) \,,

    where each component is associated with an argument and can itself be
    represented as a tuple. The first item in each such tuple is an integer
    encoding the position of the argument in the signature; the second item
    can be represented as a key-value pair where the key is in the collection
    :math:`\{ "positional", "default", "arbitrary", "keyword" \}` and the
    value is a string representing the name of the argument; the third item
    is a Python type representing the type of the argument.

    Ideally, what we would like to do is mapping each positional and keyword
    input of the target `fx.Node` to a unique integer index representing its
    position in the `fx.Node`'s signature.
    TODO: at the moment, we do not support this mapping; to allow future
          extensibility, however, I also note the position in which each
          positional argument appears in the (symbolic) call.
    """

    # split `fx.Node` from non-`fx.Node` inputs (https://stackoverflow.com/a/12135169)
    fxnode_args, other_args = [], []
    for i, a in enumerate(n.args):
        (fxnode_args if isinstance(a, fx.Node) else other_args).append((i, a))

    fxnode_kwargs, other_kwargs = {}, {}
    for k, v in n.kwargs.items():
        (fxnode_kwargs if isinstance(v, fx.Node) else other_kwargs)[k] = v

    return fxnode_args, other_args, fxnode_kwargs, other_kwargs


_undefined_eps = torch.Tensor([float('nan')])


def _undefined_eps_propagation(n: fx.Node,
                               m: Union[None, nn.Module],
                               *args,
                               **kwargs) -> torch.Tensor:
    """Return an *undefined quantum*.

    Operations that use floating-point parameters, such as `nn.Linear`,
    `nn.ConvNd`, and `nn.BatchNormNd` PyTorch `nn.Module`s do not return
    fake-quantised tensors.
    """
    return _undefined_eps


def _tolerance_eps_propagation(n: fx.Node,
                               m: Union[None, nn.Module],
                               tolerance: float = 1e-8,
                               *args,
                               **kwargs) -> torch.Tensor:
    """Compute the quantum associated with the output of a `torch.fx` node.

    This function can be applied to single-input as well as to multi-input
    operations. In the case of single-input operations, the output quantum
    will be equal to the input quantum; therefore, if the input is not
    fake-quantised (i.e., `eps_ins == [float('nan')]`), also the output will
    be marked as not fake-quantised (`eps_out = float('nan')`).

    The case of multi-input operations is meant to handle variadic operations
    such as additions and concatenations, where the semantic of the result
    (i.e., the quantum of the fake-quantised array) should coincide with the
    semantic of all the inputs to the operation. In these cases, the semantics
    of the inputs should already be coherent (i.e., all the quanta of the
    input fake-quantised arrays should coincide); we also allow for an
    approximated semantic equivalence by means of a non-negative tolerance
    parameter.

    In this last case, we verify that all the :math:`N` input quanta lie
    withing a given (either defined by the user or by the caller function)
    tolerance from each other. The computational cost of this verification
    is quadratic in :math:`N` (there are :math:`N * (N - 1) / 2` possible
    point pairs), but we assume that in most computational graphs :math:`N`
    will be sufficiently small not to notice this cost.
    """
    if tolerance < 0.0:
        raise ValueError("[QuantLib] Epsilon annotation: tolerance must be non-negative.")

    # unpack the `fx.Node`'s inputs and search the quanta of the fake-quantised ones
    fxnode_args, other_args, fxnode_kwargs, other_kwargs = _unpack_fxnode_inputs(n)
    assert len([v for v in fxnode_kwargs.values() if 'eps' in v.meta.keys()]) == 0, "[QuantLab] Processing fake-quantised arrays as keyword arguments is not supported."
    eps_ins = [input_.meta['eps'] for pos, input_ in fxnode_args if quantlib.editing.fx.annotations.EpsAnnotator.is_eps_annotated(input_)]

    # canonicalise input
    eps_ins = torch.Tensor(eps_ins)

    if torch.any(eps_ins.isnan()):  # I can't compute a numerical annotation
        eps_out = _undefined_eps
    else:
        abs_diffs = torch.abs(eps_ins[:, None] - eps_ins)  # compute pair-wise differences
        if torch.any(tolerance < abs_diffs):  # I can not disambiguate the epsilon annotation
            eps_out = _undefined_eps
        else:
            eps_out = torch.Tensor([eps_ins[0]])

    return eps_out


def _default_fqmodules_eps_propagation(n: fx.Node,
                                       m: nn.Module,
                                       *args,
                                       **kwargs) -> torch.Tensor:

    # unpack the `fx.Node`'s inputs and search the quanta of the fake-quantised ones
    fxnode_args, other_args, fxnode_kwargs, other_kwargs = _unpack_fxnode_inputs(n)
    assert len([v for v in fxnode_kwargs.values() if 'eps' in v.meta.keys()]) == 0, "[QuantLab] Processing fake-quantised arrays as keyword arguments is not supported."
    eps_ins = [input_.meta['eps'] for pos, input_ in fxnode_args if quantlib.editing.fx.annotations.EpsAnnotator.is_eps_annotated(input_)]

    # I assume that the generic `eps_out` function takes the quanta of the fake-quantised inputs as its first positional arguments
    args = [eps for eps in eps_ins] + [a for a in args]

    try:
        eps_out = m.eps_out(*args, **kwargs)
    except AttributeError:
        raise AttributeError("[QuantLab] `nn.Module` {} does not support epsilon propagation.".format(type(m)))

    return eps_out


def _adaptiveavgpoolnd_eps_propagation(n: fx.Node,
                                       m: nn.Module,
                                       *args,
                                       **kwargs) -> torch.Tensor:

    if quantlib.editing.fx.annotations.ShapeAnnotator.is_shape_annotated(n):

        fxnode_args, other_args, fxnode_kwargs, other_kwargs = _unpack_fxnode_inputs(n)
        assert len([v for v in fxnode_kwargs.values() if 'eps' in v.meta.keys()]) == 0, "[QuantLab] Processing fake-quantised arrays as keyword arguments is not supported."
        assert len(fxnode_args) == 1
        p = fxnode_args[0][1]  # unique predecessor

        if quantlib.editing.fx.annotations.ShapeAnnotator.is_shape_annotated(p):
            if n.meta['tensor_meta'].shape == p.meta['tensor_meta'].shape:
                eps_out = p.meta['eps']
            else:
                eps_out = _undefined_eps
        else:
            eps_out = _undefined_eps

    else:
        eps_out = _undefined_eps

    return eps_out


EpsPropagationSpec = collections.namedtuple('EpsPropagationSpec', ['function', 'args', 'kwargs'])


_module_2_epspec = {
    nn.ReLU: EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 0.0}),  # TODO: I assume that zero is a valid quantisation level (e.g., the quantum of {-0.33, 0.17, 0.67} is 0.5, but the quantum of ReLU({-0.33, 0.17, 0.67}) = {0.0, 0.17, 0.67} is 0.01).
    nn.Identity: EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 0.0}),
    nn.MaxPool1d: EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 0.0}),
    nn.MaxPool2d: EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 0.0}),
    nn.MaxPool3d: EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 0.0}),
    nn.AdaptiveAvgPool2d: EpsPropagationSpec(function=_adaptiveavgpoolnd_eps_propagation, args=[], kwargs={'tolerance': 0.0}),  # TODO: This must be corrected: it just works if the effective spatial support has size 1x1.
}


_module_2_epspec.update({m: EpsPropagationSpec(function=_default_fqmodules_eps_propagation, args=[], kwargs={}) for m in quantlib.algorithms.ana.ANAClasses})


_method_2_epspec = {
    'view': EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 0.0}),
    'add': EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 1e-8}),
    'concat': EpsPropagationSpec(function=_tolerance_eps_propagation, args=[], kwargs={'tolerance': 1e-8}),
}
