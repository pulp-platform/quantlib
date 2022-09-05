from typing import Optional, Tuple, List, Union, Literal

import pandas as pd
import torch
from torch import nn, fx

from quantlib.editing.fx.util import module_of_node, named_module_nodes
from quantlib.editing.fx.passes import FxPass, ModifySequentialPatternPass, ShapePropPass, CountMACsPass, SequentialPass, MemoryUsagePass, CollectPropertiesPass, AnnotateEpsPass
from quantlib.algorithms.pact import PACTIntegerAdd
from .bb_util import BB_symbolic_trace, find_layer_sets, partition_dict

from quantlib.algorithms.bb import BBAct, BBConv2d, BBLinear, BBGateController, BBBOPComplexityRegularizer, BBLatencyRegularizer, BBSplitLatencyRegularizer, BBMultiLayerController, BB_attach_gates_individual, BB_attach_gates_shared, BB_attach_gates_individual_best_latency
from quantlib.algorithms.bb.bb_ops import _BB_LINOPS
from quantlib.algorithms.bb.generate_bench_spec import import_bench_results, ident_of_layer


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

class ReadLatencyPass(FxPass):
    def __init__(self, latency_spec_file : str):
        self.latency_dict = import_bench_results(latency_spec_file)
        self.latency_spec_file = latency_spec_file

    def run_pass(self, gm : fx.GraphModule):
        for layer_name, node, module in named_module_nodes(gm):
            ident = ident_of_layer(node, module)
            if ident is not None:
                try:
                    prec_dict = self.latency_dict[ident]
                except KeyError:
                    print(f"Warning: Layer {layer_name} has identifier, but was not found in latency spec file {self.latency_spec_file}... Something is about to break!")
                    prec_dict = None
                node.meta['latency'] = prec_dict
                node.meta['max_latency'] = self.latency_dict['max_latency']
        return gm

class BBControllerPrepPass(FxPass):
    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size], latency_spec : Optional[str] = None):
        super(BBControllerPrepPass, self).__init__()
        # to know #MACs for each layer we must first know the input shapes for
        # conv layers
        self.register_subpass("shape_prop", ShapePropPass(shape_in))
        self.register_subpass("count_macs", CountMACsPass())
        self.register_subpass("memory", MemoryUsagePass())
        if latency_spec is not None:
            self.register_subpass("latency", ReadLatencyPass(latency_spec))
        else:
            self.latency = None
        self.register_subpass("properties", CollectPropertiesPass())
        self.property_dict = None


    def run_pass(self, gm : fx.GraphModule):
        gm = self.shape_prop.apply(gm)
        gm = self.count_macs.apply(gm)
        gm = self.memory.apply(gm)
        if self.latency is not None:
            gm = self.latency.apply(gm)
        gm = self.properties.apply(gm)

        max_macs = max(v['macs'] for v in self.properties.prop_dict.values() if 'macs' in v.keys())
        self.property_dict = self.properties.prop_dict.copy()
        for v in self.property_dict.values():
            v.update({'max_macs':max_macs})

        return gm



class BBActConvControllerInitPass(FxPass):


    def __init__(self, shape_in : Union[Tuple[int], List[int], torch.Size], gate_init : float = 2., input_prec : int = 8, joint_distribution : bool = False, shared_gates : bool = False, target : Literal["bops", "latency"] = "bops", latency_spec_file : Optional[str] = None, init_best_latency_gates : bool = False, split : str = None):
        super(BBActConvControllerInitPass, self).__init__()
        assert target in ["bops", "latency"], f"BBActConvControllerInitPass expected parameter 'target' with value 'bops' or 'latency', got '{target}'"
        if target == "latency":
            assert latency_spec_file is not None, f"For target=={target}, BBActConvControllerInitPass needs a latency spec file!"
            assert not shared_gates, f"BBActConvControllerInitPass: for target=={target}, shared_gates is not supported!"
        else:
            assert not init_best_latency_gates, f"For target=={target}, BBActConvController requires init_best_latency_gates==False!"
        self.target = target
        self.register_subpass("prep", BBControllerPrepPass(shape_in, latency_spec_file))
        self.input_prec = input_prec
        self.joint_distribution = joint_distribution
        self.gate_init = gate_init
        self.shared_gates = shared_gates
        self.init_best_latency_gates = init_best_latency_gates
        if shared_gates:
            self.attach_gate_fn = BB_attach_gates_shared
        elif init_best_latency_gates:
            self.attach_gate_fn = BB_attach_gates_individual_best_latency
        else:
            self.attach_gate_fn = BB_attach_gates_individual
        self.split = split


    def run_pass(self, gm : fx.GraphModule):
        gm = self.prep(gm)
        prop_dict = self.prep.property_dict
        layer_pairs = find_layer_sets(gm)

        prop_dicts_partitioned = partition_dict(gm, prop_dict, layer_pairs)

        for pd in prop_dicts_partitioned:
            if self.target == "bops":
                t_reg = BBBOPComplexityRegularizer(self.joint_distribution, self.input_prec)
            else:
                if not self.split:
                    t_reg = BBLatencyRegularizer(self.input_prec)
                else:
                    t_reg = BBSplitLatencyRegularizer(self.input_prec, self.split)
            # TODO add memory regularizer
            gate_init_kwargs = {'gate_init':self.gate_init}
            if self.init_best_latency_gates:
                gate_init_kwargs.update({'input_prec': self.input_prec})
            ctrl = BBMultiLayerController(pd, [t_reg], self.attach_gate_fn, gate_init_kwargs)

        return gm
