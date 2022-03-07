from typing import Optional, Tuple, List, Union

import torch
from torch import nn, fx

from quantlib.editing.fx.util import module_of_node
from quantlib.editing.fx.passes import FxPass, ModifySequentialPatternPass, ShapePropPass, CountMACsPass, SequentialPass, MemoryUsagePass, CollectPropertiesPass, AnnotateEpsPass
from quantlib.algorithms.pact import PACTIntegerAdd
from .bb_util import BB_symbolic_trace

from quantlib.algorithms.bb import BBAct, BBConv2d, BBLinear, BBGateController, BBBOPComplexityRegularizer, BBMultiLayerController, BB_attach_gates_individual
from quantlib.algorithms.bb.bb_ops import _BB_LINOPS


__all__ = ["BBAttachControllersPass",
           "BBControllerInitPass",
           "BBActConvControllerInitPass"]

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



class BBAttachControllersPass(FxPass):

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

    def __init__(self, max_macs : Optional[int] = None, macs_dict : Optional[dict] = None, gate_init : float = 2.):
        self.max_macs = max_macs
        self.macs_dict = macs_dict
        self.gate_init = gate_init

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
            if isinstance(m, (BBAct, PACTIntegerAdd)):
                return node
            elif isinstance(m, self._PASSTHRU_LAYERS):
                return self.find_prev_act(gm, node.all_input_nodes[0])

        return None

    def run_pass(self, gm : fx.GraphModule):
        for node in gm.graph.nodes:
            if node.op == "call_module" and isinstance(module_of_node(gm, node), tuple(_BB_LINOPS)):
                module = module_of_node(gm, node)
                if "bb_gates" not in dict(module.named_parameters()):
                    scale_factor = self.macs_dict[module]/self.max_macs
                    #print(f"Scale factor of conv/linear node {node.name}: {scale_factor}")
                    #print(f"Module repr: {module}")

                    ctrl = BBGateController(module, scale_factor, self.gate_init)
                    assert len(module.gate_ctrls) == 1, "OOPS, only single gate controller per layer supported by BBAttachControllersPass"
                    maybe_act = self.find_prev_act(gm, node.all_input_nodes[0])
                    if maybe_act:
                        am = module_of_node(gm, maybe_act)
                        # label the conv layer with the associated activation
                        module.gate_ctrls[0].linked_layer = maybe_act.target
                        # if we have an integerAdd node, the activation we are
                        # looking for is its output activation
                        if isinstance(am, PACTIntegerAdd):
                            am = am.act_out
                            module.gate_ctrls[0].linked_layer += ".act_out"

                        if isinstance(am, BBAct): #`am` may be a regular PACTAct...
                            if len(am.gate_ctrls):
                                # if there is already a gate_controller registered,
                                # we add the scale factor to the existing one. Note
                                # that this can lead to scale factors > 1!
                                # this can happen if one activation's output is
                                # used by multiple linear operators
                                assert len(am.gate_ctrls) == 1, "OOPS, only single gate controller per layer supported by BBAttachControllersPass"
                                am.gate_ctrls[0].loss_scale += scale_factor
                            else:
                                actrl = BBGateController(am, scale_factor, self.gate_init)
        return gm

class BBControllerInitPass(FxPass):
    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size], gate_init : float = 2.):
        super(BBControllerInitPass, self).__init__()
        passes = []
        # to know #MACs for each layer we must first know the input shapes for
        # conv layers
        self.register_subpass("shape_prop", ShapePropPass(shape_in))
        self.register_subpass("count_macs", CountMACsPass())
        self.register_subpass("attach_controllers", BBAttachControllersPass())



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

class BBControllerPrepPass(FxPass):
    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size], n_levels_in : int = 256):
        super(BBControllerPrepPass, self).__init__()
        # to know #MACs for each layer we must first know the input shapes for
        # conv layers
        self.register_subpass("shape_prop", ShapePropPass(shape_in))
        self.register_subpass("count_macs", CountMACsPass())
        self.register_subpass("memory", MemoryUsagePass())
        self.register_subpass("properties", CollectPropertiesPass())
        self.property_dict = None


    def run_pass(self, gm : fx.GraphModule):
        gm = self.shape_prop.apply(gm)
        gm = self.count_macs.apply(gm)
        gm = self.memory.apply(gm)
        gm = self.properties.apply(gm)

        max_macs = max(v['macs'] for v in self.properties.prop_dict.values() if 'macs' in v.keys())
        self.property_dict = self.properties.prop_dict.copy()
        for v in self.property_dict.values():
            v.update({'max_macs':max_macs})

        return gm

class BBActConvControllerInitPass(FxPass):

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

    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size], gate_init : float = 2., input_prec : int = 8, joint_distribution : bool = False):
        super(BBActConvControllerInitPass, self).__init__()
        self.register_subpass("prep", BBControllerPrepPass(shape_in, 2**input_prec))
        self.input_prec = input_prec
        self.joint_distribution = joint_distribution
        self.gate_init = gate_init

    def find_prev_act(self, gm : fx.GraphModule, node : fx.Node):
        if node.op in ["call_method", "call_function"]:
            print(f"BBAttachControllersPass: find_prev_act ignoring node {node.op}({node.target}) and continuing to node {node.all_input_nodes[0]}! If this is not correct, go and fix the code :^)")
            return self.find_prev_act(gm, node.all_input_nodes[0])
        elif node.op == "call_module":
            m = module_of_node(gm, node)
            if isinstance(m, (BBAct, PACTIntegerAdd)):
                return node
            elif isinstance(m, self._PASSTHRU_LAYERS):
                return self.find_prev_act(gm, node.all_input_nodes[0])

        return None

    def find_layer_sets(self, gm : fx.GraphModule):
        layer_pairs = []
        # we need to check if the same layer has already been found as the same
        # activation may lead into multiple linear operators

        for node in gm.graph.nodes:
            if node.op == "call_module" and isinstance(module_of_node(gm, node), tuple(_BB_LINOPS)):
                cur_layer_pair = [node.target]
                maybe_act = self.find_prev_act(gm, node.all_input_nodes[0])
                if maybe_act is not None:
                    am = module_of_node(gm, maybe_act)
                    act_target = maybe_act.target
                    if isinstance(maybe_act, PACTIntegerAdd):
                        act_target += ".act_out"
                    cur_layer_pair = [act_target] + cur_layer_pair
                layer_pairs.append(cur_layer_pair)
        return layer_pairs

    def run_pass(self, gm : fx.GraphModule):
        gm = self.prep(gm)
        prop_dict = self.prep.property_dict
        layer_pairs = self.find_layer_sets(gm)

        def partition_dict(d : dict, s : list):
            dicts_out = []
            for l_set in s:
                cur_dict = {k:[gm.get_submodule(k), d[k]] for k in l_set}
                dicts_out.append(cur_dict)
            return dicts_out

        prop_dicts_partitioned = partition_dict(prop_dict, layer_pairs)

        for pd in prop_dicts_partitioned:
            bop_reg = BBBOPComplexityRegularizer(self.joint_distribution, self.input_prec)
            # TODO add memory regularizer
            ctrl = BBMultiLayerController(pd, [bop_reg], BB_attach_gates_individual, {'gate_init':self.gate_init})

        return gm


