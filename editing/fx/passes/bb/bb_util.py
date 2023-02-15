from functools import partial
from torch import nn, fx
from ...util.tracing import LeafTracer, custom_symbolic_trace
from ...util import module_of_node
from ..pact.pact_util import PACT_OPS
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.bb.bb_ops import _BB_LINOPS
from quantlib.algorithms.pact import PACTIntegerAdd

__all__ = ["BB_OPS",
           "BBTracer",
           "BB_symbolic_trace",
           "find_prev_act",
           "find_layer_sets",
           "partition_dict"]

BB_OPS = set([BBAct,
              BBConv2d,
              BBLinear])

BBTracer = LeafTracer(leaf_types=list(BB_OPS | PACT_OPS))
BB_symbolic_trace = partial(custom_symbolic_trace, tracer=BBTracer)


# layers which we want to ignore when searching for an activation preceding
# a linear operator
_PASSTHRU_LAYERS = (nn.AdaptiveAvgPool1d,
                    nn.AdaptiveAvgPool2d,
                    nn.AvgPool1d,
                    nn.AvgPool2d,
                    nn.AdaptiveMaxPool1d,
                    nn.AdaptiveMaxPool2d,
                    nn.MaxPool1d,
                    nn.MaxPool2d,
                    nn.Flatten,
                    nn.Dropout)

def find_prev_act(gm : fx.GraphModule, node : fx.Node):
    if node.op in ["call_method", "call_function"]:
        print(f"find_prev_act ignoring node {node.op}({node.target}) and continuing to node {node.all_input_nodes[0]}! If this is not correct, go and fix the code :^)")
        return find_prev_act(gm, node.all_input_nodes[0])
    elif node.op == "call_module":
        m = module_of_node(gm, node)
        if isinstance(m, (BBAct, PACTIntegerAdd)):
            return node
        elif isinstance(m, _PASSTHRU_LAYERS):
            return find_prev_act(gm, node.all_input_nodes[0])

    return None

def find_layer_sets(gm : fx.GraphModule):
    layer_pairs = []
    # we need to check if the same layer has already been found as the same
    # activation may lead into multiple linear operators

    for node in gm.graph.nodes:
        if node.op == "call_module" and isinstance(module_of_node(gm, node), tuple(_BB_LINOPS)):
            cur_layer_pair = [node.target]
            maybe_act = find_prev_act(gm, node.all_input_nodes[0])
            if maybe_act is not None:
                am = module_of_node(gm, maybe_act)
                act_target = maybe_act.target
                if isinstance(am, PACTIntegerAdd):
                    act_target += ".act_out"
                    am = am.act_out
                cur_layer_pair = [act_target] + cur_layer_pair
            layer_pairs.append(cur_layer_pair)
    return layer_pairs


def partition_dict(gm : fx.GraphModule, d : dict, s : list):
    dicts_out = []
    for l_set in s:
        cur_dict = {k:[gm.get_submodule(k), d[k]] for k in l_set}
        dicts_out.append(cur_dict)
    return dicts_out
