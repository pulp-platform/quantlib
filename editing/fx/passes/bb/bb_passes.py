from typing import Optional, Tuple, List, Union

import torch
from torch import nn, fx

from quantlib.editing.fx.util import module_of_node
from quantlib.editing.fx.passes import FxPass, ModifySequentialPatternPass, ShapePropPass, CountMACsPass, SequentialPass
from .bb_util import BB_symbolic_trace

from quantlib.algorithms.bb import BBAct, BBConv2d, BBLinear, BBGateController



__all__ = ["BBAttachControllersPass",
           "BBControllerInitPass"]

_ALL_DEF_ARGS = {"precs": [2,4,8],
                 "hc_stretch": 1.2,
                 "hc_T": 0.5,
                 "init_clip": "max"}

_CONV_DEF_ARGS = _ALL_DEF_ARGS.copy()
_CONV_DEF_ARGS.update({"in_channels": 1,
                       "out_channels": 1,
                       "kernel_size": 1,
                       "quantize": "per_channel"})

_LIN_DEF_ARGS = _ALL_DEF_ARGS.copy()
_LIN_DEF_ARGS.update({"in_features": 1,
                      "out_features": 1,
                      "quantize": "per_channel"})

_ACT_DEF_ARGS = _ALL_DEF_ARGS.copy()
_ACT_DEF_ARGS.update({"act_kind": "relu",
                      "signed": False,
                      "learn_clip": False})



class BBAttachControllersPass(SequentialPass):
    def __init__(self, max_macs : Optional[int] = None, macs_dict : Optional[dict] = None):
        passes = []
        for p, n in [((BBAct(**_ACT_DEF_ARGS), BBConv2d(**_CONV_DEF_ARGS)), "ACT_CONV2D "), ((BBConv2d(**_CONV_DEF_ARGS),), "CONV2D"), ((BBAct(**_ACT_DEF_ARGS), BBLinear(**_LIN_DEF_ARGS)), "ACT_LINEAR"), ((BBLinear(**_LIN_DEF_ARGS),), "LINEAR")]:
            pp = nn.Sequential(*p)
            passes.append(ModifySequentialPatternPass(pp, BB_symbolic_trace, self.attach_controller, f"_QL_ATTACH_BB_CTRL_{n}", get_max_macs=self.get_max_macs, get_macs_dict=self.get_macs_dict))
        super(BBAttachControllersPass, self).__init__(*passes, name_prefix="_ATTACH_BB_CTRLS_PASS")
        self.max_macs = max_macs
        self.macs_dict = macs_dict

    def set_max_macs(self, macs):
        self.max_macs = macs

    def get_max_macs(self):
        return self.max_macs

    def set_macs_dict(self, d):
        self.macs_dict = d

    def get_macs_dict(self):
        return self.macs_dict

    @staticmethod
    def attach_controller(modules, get_max_macs, get_macs_dict):
        act = None
        conv = modules[0]
        if len(modules) == 2:
            # act + conv modules
            act = modules[0]
            conv = modules[1]

        macs_dict = get_macs_dict()
        if "bb_gates" not in dict(conv.named_parameters()):
            macs = macs_dict[conv]
            max_macs = get_max_macs()
            scale_factor = macs/max_macs
            ctrl = BBGateController(conv, scale_factor)

        if act is not None and "bb_gates" not in dict(act.named_parameters()):
            ctrl = BBGateController(act, scale_factor)

class BBAttachControllersPassFixed(FxPass):

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
                        nn.Flatten)

    def __init__(self, max_macs : Optional[int] = None, macs_dict : Optional[dict] = None):
        self.max_macs = max_macs
        self.macs_dict = macs_dict

    def set_max_macs(self, macs):
        self.max_macs = macs

    def get_max_macs(self):
        return self.max_macs

    def set_macs_dict(self, d):
        self.macs_dict = d

    def get_macs_dict(self):
        return self.macs_dict

    def find_prev_act(self, gm : fx.GraphModule, node : fx.Node):
        if node.op in ["call_method", "call_function"]:
            print(f"BBAttachControllersPass: find_prev_act ignoring node {node.op}({node.target}) and continuing to node {node.all_input_nodes[0]}! If this is not correct, go and fix the code :^)")
            return self.find_prev_act(gm, node.all_input_nodes[0])
        elif node.op == "call_module":
            m = module_of_node(gm, node)
            if isinstance(m, BBAct):
                return node
            elif isinstance(m, self._PASSTHRU_LAYERS):
                return self.find_prev_act(gm, node.all_input_nodes[0])

        return None

    def run_pass(self, gm : fx.GraphModule):
        for node in gm.graph.nodes:
            if node.op == "call_module" and isinstance(module_of_node(gm, node), (BBConv2d, BBLinear)):
                module = module_of_node(gm, node)
                if "bb_gates" not in dict(module.named_parameters()):
                    scale_factor = self.macs_dict[module]/self.max_macs
                    print(f"Scale factor of conv/linear node {node.name}: {scale_factor}")
                    print(f"Module repr: {module}")
                    ctrl = BBGateController(module, scale_factor)
                    maybe_act = self.find_prev_act(gm, node.all_input_nodes[0])
                    if maybe_act:
                        am = module_of_node(gm, maybe_act)
                        if am.gate_ctrl:
                            # if there is already a gate_controller registered,
                            # we add the scale factor to the existing one. Note
                            # that this can lead to scale factors > 1!
                            # this can happen if one activation's output is
                            # used by multiple linear operators
                            print(f"Scale factor of act node {node.name}: {scale_factor}")
                            print(f"Module repr: {am}")
                            am.gate_ctrl.loss_scale += scale_factor
                        else:
                            actrl = BBGateController(am, scale_factor)
        return gm

class BBControllerInitPass(FxPass):
    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size]):
        super(BBControllerInitPass, self).__init__()
        passes = []
        # to know #MACs for each layer we must first know the input shapes for
        # conv layers
        self.register_subpass("shape_prop", ShapePropPass(shape_in))
        self.register_subpass("count_macs", CountMACsPass())
        self.register_subpass("attach_controllers", BBAttachControllersPassFixed())



    @staticmethod
    def macs_of_module(gm : fx.GraphModule):
        macs_dict = {}
        for node in gm.graph.nodes:
            if "macs" in node.meta.keys():
                module = module_of_node(gm, node)
                macs_dict[module] = node.meta["macs"]
        return macs_dict


    def run_pass(self, gm : fx.GraphModule):
        gm = self.shape_prop.apply(gm)
        gm = self.count_macs.apply(gm)
        macs_dict = self.macs_of_module(gm)
        max_macs = max(macs_dict.values())
        self.attach_controllers.set_max_macs(max_macs)
        self.attach_controllers.set_macs_dict(macs_dict)
        return self.attach_controllers.apply(gm)


