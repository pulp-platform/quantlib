from typing import Optional

import torch
import json

import numpy as np

from torch import nn, fx

from quantlib.algorithms.bb import BBAct, BBConv2d, BBLinear, _BB_LINOPS, _BB_CLASSES
from quantlib.algorithms.pact import PACTUnsignedAct, PACTAsymmetricAct
from quantlib.algorithms import Controller
from quantlib.editing.lightweight import LightweightNode
from quantlib.editing.lightweight.rules.filters import TypeFilter


def madd_of_conv(insize, c):
    madds_per_pixel = c.kernel_size[0]*c.kernel_size[1]*c.in_channels*c.out_channels//c.groups
    return madds_per_pixel * insize * insize//(c.stride[0]*c.stride[1])
def madd_of_lin(l):
    return l.in_features * l.out_features

def fx_node_of_ql_node(gm : fx.GraphModule, n : LightweightNode):
    for fxn in gm.graph.nodes:
        if fxn.op == "call_module" and fxn.target == n.name:
            return fxn
    return None

# different policies can be used to attach gates to layers - multiple layers may share the same gates in the future
def BB_attach_gates_individual(layer_dict : dict, gate_init : float = 2.):
    for l in layer_dict.values():
        layer = l[0]
        if isinstance(layer, tuple(_BB_CLASSES)) and layer.bb_gates is None:
            n_precs = len(layer.precs)
            gates = nn.Parameter(torch.full((n_precs-1,), gate_init))
            layer.bb_gates = gates


class BBGateController:
    # A controller which calculates the loss term for a given BB layer
    # according to its complexity. The complexity is not calculated by the
    # controller itself, but instead by the entity attaching it to the layer
    # and is encoded in the "loss_scale" parameter.
    def __init__(self, layer : nn.Module, loss_scale : Optional[float] = None, gate_init : float = 2.):
        # 0. Sanity check
        assert isinstance(layer, (BBConv2d, BBLinear, BBAct)), "Layer of BBGateController must be BBAct, BBLinear, BBConv1d or BBConv2d!"

        # 1. set up fields
        self.layer = layer
        self.loss_scale = 1. if loss_scale is None else loss_scale

        # 2. set up gating parameters of conv and act layer
        n_precs = len(layer.precs)

        assert "bb_gates" not in dict(layer.named_parameters()), "Expected linear layer to not have 'bb_gates' parameter!"
        gates = nn.Parameter(torch.full((n_precs-1,), gate_init))
        layer.bb_gates = gates
        layer.register_gate_ctrl(self)
        self.gates = gates
        # for conv layers, this contains the name of the preceding activation layer
        self.linked_layer = None

    def loss_term(self):
        ccdfs = self.layer.ccdf0()
        # ehehe cumprod :^)
        cumulative_ccdfs = torch.cumprod(ccdfs, 0)
        gated_precs = torch.tensor(self.layer.precs[1:], device=cumulative_ccdfs.device)
        loss_term = torch.sum(cumulative_ccdfs * gated_precs)

        return self.loss_scale * loss_term

class BBMultiLayerController:
    # a controller which takes a variable number of layers (two controllers
    # must not overlap in the layers they manage!!) and can register a variable
    # number of regularizers
    # layer_dict should be a dict of form:
    # {layer_name : [layer, {property1_name : property1_value, property2_name :
    # property2_value, ...}]}
    # for many networks, the layer lists and properties will have to be
    # collected by FX passes
    def __init__(self, layer_dict : dict, regularizers : list, gate_init_fun : callable, gate_init_kwargs):
        self.layers = layer_dict
        self.regularizers = regularizers
        for r in self.regularizers:
            r.register_layers(self.layers)

        gate_init_fun(layer_dict, **gate_init_kwargs)
        for l in self.layers.values():
            l[0].register_gate_ctrl(self)

    def loss_term(self):
        lt = 0.
        for r in self.regularizers:
            lt = lt + r.loss_term()

        return lt

class BBRegularizer:
    def register_layers(self, layers : dict, *args, **kwargs):
        raise NotImplementedError

    def loss_term(self):
        raise NotImplementedError


class BBBOPComplexityRegularizer(BBRegularizer):
    def __init__(self, joint_distribution : bool = False, input_prec : Optional[int] = 8):
        self.input_prec = input_prec
        self.joint_distribution = joint_distribution

    def register_layers(self, layers : dict):
        assert len(layers) in [1,2], f"BBBOPComplexityRegularizer must takes 1 or 2 layers - got {len(layers)} instead!"
        print(f"BBBOPComplexityRegularizer for layers {[k for k in layers.keys()]} registered")
        op_layer_and_props = [l for l in layers.values() if isinstance(l[0], tuple(_BB_LINOPS))][0]
        self.op_layer = op_layer_and_props[0]
        self.loss_scale = op_layer_and_props[1]['macs']/op_layer_and_props[1]['max_macs']
        act_layer = None
        if len(layers) != 1:
            act_layer_and_props = [v for k, v in layers.items() if v[0] is not self.op_layer][0]
            act_layer = act_layer_and_props[0]
        self.act_layer = act_layer

    def loss_term(self):
        # this loss term calculates the joint probability distribution of
        # activation and linop precision gates to find the expected number of
        # BOPs. this is DIFFERENT from the paper which treats them separately.
        # for example, if a convolutional layer has only 2 bits with high
        # probability, having 8-bit input activations has less impact on the
        # total BOP count than if it has 8. by summing over the joint
        # distribution of the precision gates, we can account for this.

        # helper to get tensors on the right device
        def move(t : torch.Tensor):
            return t.type_as(self.op_layer.weight.data)

        one = move(torch.ones((1,)))

        # no activation -> this must be the input layer
        if self.act_layer is None:
            act_precs = move(torch.tensor([self.input_prec]))
            act_ccdfs = one
        elif isinstance(self.act_layer, (PACTUnsignedAct, PACTAsymmetricAct)):
            act_precs = move(torch.tensor([int(np.ceil(np.log2(self.act_layer.n_levels)))]))
            act_ccdfs = one
        else:
            act_precs = move(torch.tensor(self.act_layer.precs[1:]))
            act_ccdfs = self.act_layer.ccdf0()

        op_ccdfs = self.op_layer.ccdf0()
        cum_op_ccdfs = torch.cumprod(op_ccdfs, 0)
        cum_act_ccdfs = torch.cumprod(act_ccdfs, 0)
        op_precs = move(torch.tensor(self.op_layer.precs[1:]))
        if self.joint_distribution:
            # now we can produce the joint distribution...
            joint_ccdf = torch.outer(cum_act_ccdfs, cum_op_ccdfs)
            # ...and the "BOP per MAC" matrix of the same dimension...
            bop_per_mac = torch.outer(act_precs, op_precs)
            # and use the former to weight the latter and sum everything up to get
            # a measure of "expected BOPs"
            loss_term = torch.sum(joint_ccdf * bop_per_mac)
        else:
            loss_term = torch.sum(cum_op_ccdfs * op_precs)
            if self.act_layer is not None and isinstance(self.act_layer, BBAct):
                loss_term += torch.sum(cum_act_ccdfs * act_precs)

        #finally, weight everything by #MACs/#max_MACS
        return self.loss_scale * loss_term





class BBExportController(Controller):
    # a controller which exports an info .json about a BB-quantized network.
    # The .json contains the number of levels for each layer as well as info
    # such as the total number of BOPs and equivalent number of bits
    def __init__(self, nodes_list : list, export_file : str, input_bitwidth : int = 8, net : Optional[fx.GraphModule] = None):
        self.nodes_list = nodes_list
        self.export_file = export_file
        self.input_bitwidth = input_bitwidth
        # if `net` is an instance of DataParallel, we have to strip 'module'
        # from node names
        net_is_dp = False
        if isinstance(net, nn.DataParallel):
            net_is_dp = True
            net = net.module
        self.net_is_dp = net_is_dp
        self.net = net

    @staticmethod
    def get_modules(*args, **kwargs):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def step_pre_training_epoch(self, *args, **kwargs):
        pass

    def step_pre_training_batch(self, *args, **kwargs):
        pass

    def fix_node_name(self, name : str):
        if self.net_is_dp:
            return name.lstrip('module.')
        return name

    def step_pre_validation_epoch(self, *args, **kwargs):
        prec_dict = {}
        tot_bops = 0
        bops_8b = 0
        bb_filter = TypeFilter(BBLinear) | TypeFilter(BBConv2d) | TypeFilter(BBAct)
        bb_nodes = bb_filter(self.nodes_list)
        for n in bb_nodes:
            module_levels = n.module.get_n_levels()
            module_prec = int(np.ceil(np.log2(module_levels)))
            prec_dict[n.name + '$'] = module_levels

            if isinstance(n.module, tuple(_BB_LINOPS)) and self.net is not None:
                if len(n.module.gate_ctrls) == 1 and isinstance(n.module.gate_ctrls[0], BBGateController):
                    gate_ctrl = n.module.gate_ctrls[0]

                    if gate_ctrl.linked_layer is not None:
                        # get the bitwidth of the activation preceding the current
                        # linear layer
                        linked_act = [n_.module for n_ in self.nodes_list if self.fix_node_name(n_.name) == n.module.gate_ctrl.linked_layer][0]
                        n_levels = linked_act.get_n_levels() if isinstance(linked_act, BBAct) else linked_act.n_levels
                        in_prec = int(np.ceil(np.log2(n_levels)))
                    else:
                        #if a linear layer has no "linked_layer" registered, it
                        #should be the input layer.
                        in_prec = self.input_bitwidth
                elif len(n.module.gate_ctrls) == 1 and isinstance(n.module.gate_ctrls[0], BBMultiLayerController):
                    gate_ctrl = n.module.gate_ctrls[0]
                    if len(gate_ctrl.layers) == 2:
                        linked_act = [v[0] for v in gate_ctrl.layers.values() if v[0] is not n.module][0]
                        n_levels = linked_act.get_n_levels() if isinstance(linked_act, BBAct) else linked_act.n_levels
                        in_prec = int(np.ceil(np.log2(n_levels)))
                    else:
                        in_prec = self.input_bitwidth
                else:
                    assert False, f"Unsupported BB Gate Controller configuration of layer {n.name}"
                n = LightweightNode(self.fix_node_name(n.name), n.module)

                fx_node = fx_node_of_ql_node(self.net, n)
                tot_bops += fx_node.meta['macs'] * in_prec * module_prec
                bops_8b += fx_node.meta['macs'] * 64

        bop_ratio = tot_bops/bops_8b
        out_dict = {
            'total_bops' : tot_bops,
            '8b_bops' : bops_8b,
            'bop_ratio': bop_ratio,
            'eff_bw': 8 * np.sqrt(bop_ratio),
            'layer_levels' : prec_dict}

        with open(self.export_file, 'w') as outfile:
            json.dump(out_dict, outfile, indent=4)
