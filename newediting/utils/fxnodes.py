import torch.fx as fx
from typing import Tuple, Dict, Any


def unpack_fxnode_arguments(n: fx.Node) -> Tuple[Tuple[Any], Dict[str, Any]]:

    args   = n.args
    kwargs = n.kwargs

    return args, kwargs
