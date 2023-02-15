from typing import Optional
from itertools import product

import torch
import json

import numpy as np
from scipy.optimize import nnls

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

def BB_attach_gates_individual_best_latency(layer_dict : dict, gate_init : float = 2., input_prec : int = 8):
    op_layer, op_name = [(l[0], k) for k, l in layer_dict.items() if isinstance(l[0], tuple(_BB_LINOPS))][0]
    op_precs = op_layer.precs
    if len(layer_dict) == 2:
        act_layer = [l[0] for l in layer_dict.values() if not isinstance(l[0], tuple(_BB_LINOPS))][0]
        act_precs = act_layer.precs
    else:
        assert len(layer_dict) == 1, f"BB_attach_gates_individual_best_latency expects to get 1 or 2 layers - got {len(layer_dict)}"
        act_layer = None
        act_precs = [input_prec]

    lat_dict = layer_dict[op_name][1]['latency']
    best_config, _ = min(((k,v) for k,v in lat_dict.items() if k[0] in act_precs), key=lambda kv: kv[1])
    op_gates_to_enable = op_precs.index(best_config[1])
    op_gates = torch.full((len(op_precs)-1,), -gate_init)

    for i in range(op_gates_to_enable):
        op_gates[i] = gate_init
    op_layer.bb_gates = nn.Parameter(op_gates)
    if act_layer is not None:
        act_gates_to_enable = act_precs.index(best_config[0])
        act_gates = torch.full((len(act_precs)-1,), -gate_init)
        for i in range(act_gates_to_enable):
            act_gates[i] = gate_init
        act_layer.bb_gates = nn.Parameter(act_gates)


def BB_attach_gates_shared(layer_dict : dict, gate_init : float = 2.):
    n_precs = []
    precs = []
    for l in layer_dict.values():
        layer = l[0]
        if isinstance(layer, tuple(_BB_CLASSES)) and layer.bb_gates is None:
            n_precs.append(len(layer.precs))
            precs.append(len(layer.precs))

    assert all(p == n_precs[0] for p in n_precs), "Unequal number of precisions in BB_attach_gates_shared... this does not make sense!"
    assert all(p == precs[0] for p in precs), "Unequal precisions in BB_attach_gates_shared... this does not make sense!"
    gates = nn.Parameter(torch.full((n_precs[0]-1,), gate_init))
    for l in layer_dict.values():
        layer = l[0]
        if isinstance(layer, tuple(_BB_CLASSES)) and layer.bb_gates is None:
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
        for n, l in self.layers.items():
            try:
                register_fn = l[0].register_gate_ctrl
            except AttributeError as e:
                print(f"BBMultiLayerController: Did not register BB gate controller for layer {n} of type {type(l[0])} as it does not seem to be a BB layer...")
                register_fn = lambda _ : None
            register_fn(self)

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
    def __init__(self, joint_distribution : bool = False, input_prec : int = 8):
        self.input_prec = input_prec
        self.joint_distribution = joint_distribution

    def register_layers(self, layers : dict):
        assert len(layers) in [1,2], f"BBBOPComplexityRegularizer must take 1 or 2 layers - got {len(layers)} instead!"
        print(f"BBBOPComplexityRegularizer for layers {[k for k in layers.keys()]} registered")
        self.op_layer_name, op_layer_and_props = [(k, l) for k, l in layers.items() if isinstance(l[0], tuple(_BB_LINOPS))][0]
        self.op_layer = op_layer_and_props[0]
        self.loss_scale = op_layer_and_props[1]['macs']/op_layer_and_props[1]['max_macs']
        #print(f"loss_scale: {self.loss_scale}, nmacs: {op_layer_and_props[1]['macs']}, max_macs: {op_layer_and_props[1]['max_macs']}")
        self.act_layer_name = None
        act_layer = None
        if len(layers) != 1:
            self.act_layer_name, act_layer_and_props = [(k, v) for k, v in layers.items() if v[0] is not self.op_layer][0]
            act_layer = act_layer_and_props[0]

        self.act_layer = act_layer

    def loss_term(self):
        # this loss term can optionally calculate the joint probability
        # distribution of activation and linop precision gates to find the
        # expected number of BOPs. this is DIFFERENT from the paper which
        # treats them separately. for example, if a convolutional layer has
        # only 2 bits with high probability, having 8-bit input activations has
        # less impact on the total BOP count than if it has 8. by summing over
        # the joint distribution of the precision gates, we can account for
        # this.

        # helper to get tensors on the right device
        def move(t : torch.Tensor):
            return t.type_as(self.op_layer.weight.data)

        one = move(torch.ones((1,)))
        # if quantization is not started yet, do not contribute a loss so the
        # gates stay at their initial value until quantization is started!
        if not self.op_layer.started:
            return torch.zeros([])
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
            #print(f"op loss term for layer {self.op_layer_name}: {loss_term}")
            if self.act_layer is not None and isinstance(self.act_layer, BBAct):
                act_loss_term = torch.sum(cum_act_ccdfs * act_precs)
                #print(f"activation loss term for layer {self.act_layer_name}: {act_loss_term}")
                loss_term += act_loss_term

        #finally, weight everything by #MACs/#max_MACS
        return self.loss_scale * loss_term


class BBLatencyRegularizer(BBRegularizer):

    def __init__(self, input_prec : int = 8):
        self.input_prec = input_prec

    def register_layers(self, layers : dict):
        assert len(layers) in [1,2], f"BBLatencyRegularizer takes 1 or 2 layers - got {len(layers)} instead!"
        print(f"BBLatencyRegularizer for layers {[k for k in layers.keys()]} registered")
        op_layer_and_props = [l for l in layers.values() if isinstance(l[0], tuple(_BB_LINOPS))][0]
        self.op_name = [k for k,v in layers.items() if isinstance(v[0], tuple(_BB_LINOPS))][0]
        if len(layers) == 2:
            self.act_name = [k for k,v in layers.items() if k != self.op_name][0]
        else:
            self.act_name = None
        self.op_layer = op_layer_and_props[0]
        self.op_layer_props = op_layer_and_props[1]
        op_precs = self.op_layer.precs
        act_layer = None
        act_precs = [self.input_prec]
        if len(layers) != 1:
            act_layer_and_props = [v for k, v in layers.items() if v[0] is not self.op_layer][0]
            act_layer = act_layer_and_props[0]
            if isinstance(act_layer, (PACTUnsignedAct, PACTAsymmetricAct)):
                act_precs = [int(np.ceil(np.log2(act_layer.n_levels)))]
            elif isinstance(act_layer, BBAct):
                act_precs = act_layer.precs
            else:
                assert False, f"BBLatencyRegularizer got incompatible activation type: {type(act_layer)}"

        # fill a matrix with the latencies for each precision configuration
        self.scales = torch.zeros(len(act_precs), len(op_precs)).type_as(self.op_layer.weight.data)
        for (y, act_prec), (x, op_prec) in product(enumerate(act_precs), enumerate(op_precs)):
            self.scales[y, x] = op_layer_and_props[1]['latency'][(act_prec, op_prec)]

        # normalize by the highest latency in the network
        self.scales = self.scales / op_layer_and_props[1]['max_latency']
        # make sure the lowest cost is 0 by shifting all costs by the minimum
        # cost
        self.scales = self.scales - self.scales.min()
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
        # if quantization is not started yet, do not contribute a loss so the
        # gates stay at their initial value until quantization is started!
        if not self.op_layer.started:
            return torch.zeros([])
        # no activation -> this must be the input layer
        if self.act_layer is None or isinstance(self.act_layer, (PACTUnsignedAct, PACTAsymmetricAct)):
            act_ccdfs = one
        else:
            act_ccdfs = torch.cat((one, self.act_layer.ccdf0()))

        # first gate is always turned on
        op_ccdfs = torch.cat((one,self.op_layer.ccdf0()))
        cum_op_ccdfs = torch.cumprod(op_ccdfs, 0)
        cum_act_ccdfs = torch.cumprod(act_ccdfs, 0)
        # we want to "isolate" the precisions; this means that not only gates
        # [0...i] should be turned on but gate i+1 should be turned off
        op_cdfs = torch.cat((1. - op_ccdfs[1:], one))
        act_cdfs = torch.cat((1. - act_ccdfs[1:], one))

        op_precs = move(torch.tensor(self.op_layer.precs))
        cum_op_ccdfs = cum_op_ccdfs * op_cdfs
        cum_act_ccdfs = cum_act_ccdfs * act_cdfs
        # now we can produce the joint distribution...
        joint_ccdf = torch.outer(cum_act_ccdfs, cum_op_ccdfs)
        # and use it to weight the normalized latencies and sum everything up to get
        # a measure of "expected latency"
        loss_term = torch.sum(joint_ccdf * move(self.scales))
        #...done!
        return loss_term

class BBSplitLatencyRegularizer(BBLatencyRegularizer):
    def __init__(self, input_prec : int, mode : str):
        self.input_prec = input_prec
        mode = mode.lower()
        assert mode in ["decompose", "marginalize"], f"BBSplitLatencyRegularizer expected 'mode' to be 'decompose' or 'marginalize - got {mode}"
        self.mode = mode

    def register_layers(self, layers : dict):
        super(BBSplitLatencyRegularizer, self).register_layers(layers)
        # the scales calculated before are factorized into the closest possible
        # additive representation if we have 2 layers. we force the optimal
        # configuration's weight and act precision penalties to 0 to make the
        # problem uniquely solvable
        if self.act_layer is not None:
            if self.mode == "decompose":
                best_act_idx, best_weight_idx = torch.nonzero(self.scales==0)[0]
                best_act_idx, best_weight_idx = int(best_act_idx), int(best_weight_idx)
                contribution_a = torch.eye(self.scales.shape[0])
                contribution_a = torch.cat([contribution_a]*self.scales.shape[1])
                contrib_a_idxs = [i for i in range(self.scales.shape[0]) if i != best_act_idx]
                contribution_a = contribution_a[:, contrib_a_idxs]
                contribution_w = [torch.zeros(self.scales.shape[0], self.scales.shape[1]) for _ in range(self.scales.shape[1])]

                for k in range(self.scales.shape[1]):
                    contribution_w[k][:, k] = 1
                contribution_w = torch.cat(contribution_w)
                contrib_w_idxs = [i for i in range(self.scales.shape[1]) if i != best_weight_idx]
                contribution_w = contribution_w[:, contrib_w_idxs]
                contrib = torch.cat([contribution_a, contribution_w], dim=1).numpy()
                penalty_target = self.scales.transpose(1,0).flatten().numpy()
                # we need nonnegative target penalties, so use SciPy's NNLS solver
                #l2_penalties, _, _, _ = np.linalg.lstsq(contrib, penalty_target)
                l2_penalties, _ = nnls(contrib, penalty_target)
                act_penalties = np.zeros(self.scales.shape[0])
                act_penalties[contrib_a_idxs] = l2_penalties[:self.scales.shape[0]-1]
                self.act_penalties = torch.tensor(act_penalties)
                wt_penalties = np.zeros(self.scales.shape[1])
                wt_penalties[contrib_w_idxs] = l2_penalties[self.scales.shape[0]-1:]
                self.wt_penalties = torch.tensor(wt_penalties)
                print(f"Weight penalties:\n{self.wt_penalties}\nAct penalties:\n{self.act_penalties}")
                wtp = torch.stack([self.wt_penalties for k in range(self.scales.shape[0])], dim=0)
                ap = torch.stack([self.act_penalties for k in range(self.scales.shape[1])], dim=1)
                print(f"Scales:\n{self.scales}\nApproximated scales:\n{wtp+ap}")
            else:
                self.act_penalties = torch.mean(self.scales, dim=1)
                self.wt_penalties = torch.mean(self.scales, dim=0)

    def loss_term(self):
        if self.act_layer is not None:
            # helper to get tensors on the right device
            def move(t : torch.Tensor):
                return t.type_as(self.op_layer.weight.data)

            one = move(torch.ones((1,)))
            # if quantization is not started yet, do not contribute a loss so the
            # gates stay at their initial value until quantization is started!
            if not self.op_layer.started:
                return torch.zeros([])
            # no activation -> this must be the input layer
            if self.act_layer is None or isinstance(self.act_layer, (PACTUnsignedAct, PACTAsymmetricAct)):
                act_ccdfs = one
            else:
                act_ccdfs = torch.cat((one, self.act_layer.ccdf0()))

            # first gate is always turned on
            op_ccdfs = torch.cat((one,self.op_layer.ccdf0()))
            cum_op_ccdfs = torch.cumprod(op_ccdfs, 0)
            cum_act_ccdfs = torch.cumprod(act_ccdfs, 0)
            # we want to "isolate" the precisions; this means that not only gates
            # [0...i] should be turned on but gate i+1 should be turned off
            op_cdfs = torch.cat((1. - op_ccdfs[1:], one))
            act_cdfs = torch.cat((1. - act_ccdfs[1:], one))
            op_precs = move(torch.tensor(self.op_layer.precs))
            cum_op_ccdfs = cum_op_ccdfs * op_cdfs
            cum_act_ccdfs = cum_act_ccdfs * act_cdfs
            loss_term = torch.sum(cum_op_ccdfs * move(self.wt_penalties))
            loss_term = loss_term + torch.sum(cum_act_ccdfs * move(self.act_penalties))
        else:
            loss_term = super(BBSplitLatencyRegularizer, self).loss_term()

        return loss_term





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


    def get_export_dict(self):
        prec_dict = {}
        lat_dict = {}
        tot_bops = 0
        bops_8b = 0
        bb_filter = TypeFilter(BBLinear) | TypeFilter(BBConv2d) | TypeFilter(BBAct)
        bb_nodes = bb_filter(self.nodes_list)
        tot_lat = 0
        lat_8b = 0
        lat_4b = 0
        min_lat = 0
        layer_cfg_dict = {}
        best_cfgs_dict = {}
        freebie_cfg_dict = {}
        freebie_levels = {}
        freebie_lat = 0
        n_freebie_layers = 0
        freebie_lat_saved = 0
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
                    if all(isinstance(r, BBLatencyRegularizer) for r in n.module.gate_ctrls[0].regularizers):
                        cur_lat_dict = n.module.gate_ctrls[0].regularizers[0].op_layer_props['latency']
                        cur_lat = cur_lat_dict[(in_prec, module_prec)]
                        better_lats = {(ap,wp):v for (ap, wp), v in cur_lat_dict.items() if ap >= in_prec and wp >= module_prec and v < cur_lat}
                        if len(better_lats):
                            freebie_cfg, freebie_lat = min(better_lats.items(), key=lambda kv:kv[1])
                            freebie_cfg_dict[n.name] = {'precs': freebie_cfg, 'latency': freebie_lat}
                            freebie_lat_saved += cur_lat - freebie_lat
                            n_freebie_layers += 1
                            freebie_levels[n.name+'$'] = int(2**freebie_cfg[1])
                            linked_act_name =  n.module.gate_ctrls[0].regularizers[0].act_name
                            if linked_act_name:
                                freebie_levels[linked_act_name+'$'] = int(2**freebie_cfg[0])
                        lat_dict[n.name] = cur_lat
                        tot_lat += cur_lat
                        lat_8b += cur_lat_dict[(8, 8)]
                        try:
                            lat_4b += cur_lat_dict[(4, 4)]
                        # some layers' input activations will be fixed to 8b
                        # and not registered in the latency dict, so take the
                        # 8/4 latency
                        except KeyError:
                            lat_4b += cur_lat_dict[(8, 4)]

                        cur_best_cfg, cur_min_lat = min(cur_lat_dict.items(), key=lambda kv:kv[1])
                        best_cfgs_dict[n.name] = {'precs': cur_best_cfg, 'latency': cur_min_lat}
                        min_lat += cur_min_lat
                else:
                    assert False, f"Unsupported BB Gate Controller configuration of layer {n.name}"
                n = LightweightNode(self.fix_node_name(n.name), n.module)
                layer_cfg_dict[n.name] = (in_prec, module_prec)
                fx_node = fx_node_of_ql_node(self.net, n)
                tot_bops += fx_node.meta['macs'] * in_prec * module_prec
                bops_8b += fx_node.meta['macs'] * 64

        bop_ratio = tot_bops/bops_8b
        out_dict = {
            'total_bops' : tot_bops,
            '8b_bops' : bops_8b,
            'bop_ratio': bop_ratio,
            'eff_bw': 8 * np.sqrt(bop_ratio),
            'layer_levels' : prec_dict,
            'layer_precs' : layer_cfg_dict}
        if len(lat_dict) != 0:
            out_dict['best_cfgs'] = best_cfgs_dict
            out_dict['freebie_cfgs'] = freebie_cfg_dict
            out_dict['freebie_levels'] = freebie_levels
            out_dict['freebie_latency_saved_abs'] = freebie_lat_saved
            out_dict['freebie_latency_saved_rel'] = freebie_lat_saved/tot_lat
            out_dict['num_freebie_layers'] = n_freebie_layers
            out_dict['latency'] = lat_dict
            out_dict['total_latency'] = tot_lat
            out_dict['8b_latency'] = lat_8b
            out_dict['4b_latency'] = lat_4b
            out_dict['min_latency'] = min_lat


        return out_dict

    def step_pre_validation_epoch(self, *args, **kwargs):
        out_dict = self.get_export_dict()
        with open(self.export_file, 'w') as outfile:
            json.dump(out_dict, outfile, indent=4)
