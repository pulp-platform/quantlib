from collections import OrderedDict
import torch.nn as nn
import torch.fx as fx

from quantlib.editing.editing.fake2true.annotation import ShapePropagator
from quantlib.editing.graphs.fx import unpack_then_split_fxnode_arguments


def verify_avgpoolnd(n: fx.Node) -> bool:

    if ShapePropagator.is_shape_annotated(n):

        # TODO: I am using this information about `AvgPoolNd` in a similar way
        #       to what I do in `EpsPropagator`. Maybe we should develop a
        #       general-purpose type-checker for `fx.Node`s.

        # unpack the `fx.Node`'s inputs
        fxnode_args, _, fxnode_kwargs, _ = unpack_then_split_fxnode_arguments(n)

        if len(fxnode_args) > 1:
            raise RuntimeError("AdaptiveAvgPoolNd expects a single argument.")

        p = next(iter(fxnode_args)).fxnode  # unique predecessor
        if ShapePropagator.is_shape_annotated(p):
            if n.meta['tensor_meta'].shape == p.meta['tensor_meta'].shape:
                state = True
            else:
                state = False
        else:
            state = False

    else:
        state = False

    return state


whitelist_call_module = OrderedDict([
    # flatten
    (nn.Flatten, []),
    # max pooling
    (nn.MaxPool1d, []),
    (nn.MaxPool2d, []),
    (nn.MaxPool3d, []),
    (nn.AdaptiveMaxPool1d, []),
    (nn.AdaptiveMaxPool2d, []),
    (nn.AdaptiveMaxPool3d, []),
    # average pooling
    (nn.AvgPool1d, [verify_avgpoolnd]),
    (nn.AvgPool2d, [verify_avgpoolnd]),
    (nn.AvgPool3d, [verify_avgpoolnd]),
    (nn.AdaptiveAvgPool1d, [verify_avgpoolnd]),
    (nn.AdaptiveAvgPool2d, [verify_avgpoolnd]),
    (nn.AdaptiveAvgPool3d, [verify_avgpoolnd]),
])


whitelist_call_method = OrderedDict([
    ('add',  []),
    ('view', []),
])


whitelist_call_function = OrderedDict([
    ('add', []),
])
