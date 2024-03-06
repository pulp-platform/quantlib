#
# export_tnn.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Joan Mihali
#
# Copyright (c) 2020-2021 ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import torch
from torch import nn
from torch.nn import Hardtanh
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import Conv1d, Conv2d
from torch.nn import Identity
from collections import namedtuple
from quantlib.algorithms.pact.pact_ops import PACTCausalConv1d, PACTConv1d, PACTConv2d, PACTLinear, PACTUnsignedAct, PACTAsymmetricAct, _PACTLinOp, _PACTActivation
from quantlib.algorithms.inq import INQConv1d, INQConv2d, INQCausalConv1d, INQLinear
from quantlib.algorithms.ste import STEActivation
from quantlib.QTensor import QTensor

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight.rules import LightweightRule as lwr
from quantlib.editing.lightweight import LightweightGraph as lwg


import numpy as np
import json
from pathlib import Path


Thresholds = namedtuple('Thresholds', 'lo hi')

pact_linear_types = [PACTConv2d, PACTConv1d, PACTLinear, PACTCausalConv1d]
inq_linear_types = [INQConv2d, INQConv1d, INQCausalConv1d, INQLinear]
linear_types = pact_linear_types + inq_linear_types
bn_types = [BatchNorm1d, BatchNorm2d]
pact_act_types = [PACTUnsignedAct, PACTAsymmetricAct]
ste_act_types = [STEActivation]
act_types = pact_act_types + ste_act_types
maxpool_types = [MaxPool1d, MaxPool2d, AdaptiveMaxPool2d]
maxpool_2d = [MaxPool2d]
adaptive_pool_types = [AdaptiveAvgPool2d, AdaptiveMaxPool2d]
avgpool_types = [AvgPool1d, AvgPool2d, AdaptiveAvgPool2d] # don't support adaptiveavgpool1d for now.
avgpool_2d = [AvgPool2d]
pool_types = maxpool_types + avgpool_types
pool_types_2d = maxpool_2d + avgpool_2d

class TernaryActivation(nn.Module):
    def __init__(self, thresh_lo : torch.Tensor, thresh_hi : torch.Tensor, n_d : int, dbg : bool = False, cutie_style_threshs : bool = False):
        super(TernaryActivation, self).__init__()
        assert torch.sum(thresh_lo % 1) == 0, "thresh_lo is not integer!"
        assert torch.sum(thresh_hi % 1) == 0, "thresh_hi is not integer!"
        assert n_d in (1,2), "Only 1D, 2D supported"
        thresh_lo = torch.unsqueeze(thresh_lo, 0)
        thresh_lo = torch.unsqueeze(thresh_lo, 2)
        thresh_hi = torch.unsqueeze(thresh_hi, 0)
        thresh_hi = torch.unsqueeze(thresh_hi, 2)
        if n_d == 2:
            thresh_lo = torch.unsqueeze(thresh_lo, 3)
            thresh_hi = torch.unsqueeze(thresh_hi, 3)

        self.thresh_lo = nn.Parameter(thresh_lo.clone().detach(), requires_grad=False)
        self.thresh_hi = nn.Parameter(thresh_hi.clone().detach(), requires_grad=False)
        self.cutie_threshs = cutie_style_threshs
        self.dbg = dbg
        self.eps = 1e-4

    def forward(self, x):
        # this will trigger even when everything is fine due to float magicks...
        if self.dbg:
            assert torch.sum(x - x.round()) < self.eps, "x is not integer in TernaryAct.forward()!"
        out = torch.ones_like(x)
        # assume NCHW tensors
        out[x<self.thresh_lo] = -1
        # CUTIE hardware uses greater-than thresholding rather than greater-or-equal
        if self.cutie_threshs:
            out[torch.logical_and(x>=self.thresh_lo, x<=self.thresh_hi)] = 0
        else:
            out[torch.logical_and(x>=self.thresh_lo, x<self.thresh_hi)] = 0
        if isinstance(out, QTensor):
            out.eps = 1.
        return out

def expand_kernel(module):
    # expand kernel of a module
    # For example, a Conv2d layer may have a "kernel_size" of 3 - this function will expand it to (3,3)
    n_dims = int(module.__class__.__name__[-2])
    k = module.kernel_size
    if not (type(k) is tuple):
        k = (k,) * n_dims
    return k

def get_threshs(nodes, prev_nodes, adaptive_kernel=None, cutie_style_threshs=False, first_layer=True):
    # returns a Thresholds tuple for a sequence of nodes:
    # nodes[0] must be a conv node
    # nodes[1] must be a BatchNorm node or an Pool node
    # nodes[2] may be a Pool Node (if pooling and layer_order=="bn_pool"), a BN
    #          Node (if pooling and layer_order=="pool_bn" or an activation
    #          node)
    # nodes[3] if present, must be an activation node
    conv_node = nodes[0].module
    assert type(conv_node) in linear_types, 'Node is not linear type'

    if type(nodes[-1].module) in act_types:
        act_node = nodes[-1].module
        if isinstance(act_node, tuple(pact_act_types)):
            eps_act = act_node.get_eps()
        else:
            eps_act = torch.ones([])
    else:
        act_node = None

    if prev_nodes is not None and type(prev_nodes[-1].module) in act_types:
        prev_act_node = prev_nodes[-1].module
    else:
        prev_act_node = None

    pooling = True
    if node_is_in(nodes[1], pool_types):
        pool_node = nodes[1]
        bn_node = nodes[2].module
    elif node_is_in(nodes[2], pool_types):
        pool_node = nodes[2]
        bn_node = nodes[1].module
    else:
        bn_node = nodes[1].module
        pooling = False
    if pooling:
        avgpool = node_is_in(pool_node, avgpool_types)

    if conv_node.bias is not None:
        bias = conv_node.bias.data
    else:
        bias = torch.zeros(conv_node.out_channels)

    if isinstance(conv_node, tuple(pact_linear_types)):
        eps_w = conv_node.get_eps_w().squeeze()
    else:
        eps_w = torch.ones([])
    if isinstance(prev_act_node, tuple(pact_act_types)):
        prev_act_eps = prev_act_node.get_eps()
    else:
        prev_act_eps = torch.ones([])

    unsigned_indicator = 1. if isinstance(act_node, PACTUnsignedAct) else 0.
    prev_unsigned = isinstance(prev_act_node, PACTUnsignedAct)
    if not first_layer and prev_unsigned:
        if isinstance(conv_node, (_PACTLinOp)):
            weights_to_sum = conv_node.weight_int
        elif isinstance(conv_node, tuple(inq_linear_types)):
            weights_to_sum = conv_node.weight_frozen
    else:
        weights_to_sum = torch.zeros(conv_node.out_channels,1,1)

    if len(weights_to_sum.shape)==4:
        dims_to_sum = (1,2,3)
    else:
        dims_to_sum = (1,2)

    bias_add = weights_to_sum.sum(dim=dims_to_sum)
    bias_add *= torch.tensor(prev_act_eps * eps_w).squeeze()
    bias_hat = bias + bias_add


    beta_hat = (bias_hat - bn_node.running_mean.data)/torch.sqrt(bn_node.running_var.data+bn_node.eps)
    gamma_hat = 1/torch.sqrt(bn_node.running_var.data+bn_node.eps)
    if bn_node.affine:
        beta_hat *= bn_node.weight.data
        beta_hat += bn_node.bias.data
        gamma_hat *= bn_node.weight.data

    if pooling and avgpool:
        if adaptive_kernel:
            pool_k = adaptive_kernel
        else:
            pool_k = expand_kernel(pool_node)
        gamma_hat *= 1.0/torch.prod(torch.tensor(pool_k, dtype=gamma_hat.dtype))

    thresh_lo = ((-0.5 + unsigned_indicator)*eps_act-beta_hat)/(gamma_hat*eps_w)
    thresh_hi = ((0.5 + unsigned_indicator)*eps_act-beta_hat)/(gamma_hat*eps_w)
    thresh_lo /= prev_act_eps
    thresh_hi /= prev_act_eps
    # if some gamma_hats/gammas are negative, the smaller than/larger than relationships are flipped there.
    # the weights in the convolution preceding the BN will be flipped for those channels, so we can simply flip the
    # thresholds. Important: flip BEFORE rounding otherwise they will be off by one :)
    if bn_node.affine:
        flip_idxs = bn_node.weight.data < 0
        thresh_hi[flip_idxs] *= -1
        thresh_lo[flip_idxs] *= -1

    thresh_lo = torch.ceil(thresh_lo)
    thresh_hi = torch.ceil(thresh_hi)

    # do this check before converting to CUTIE style thresholds - on CUTIE,
    # thresh_hi may be 1 lower than thresh_lo.
    if not torch.all(thresh_lo <= thresh_hi):
        print('All thresh_lo need to be <= thresh_hi')
        assert False
    # CUTIE's upper thresholding condition is:
    # th(x) = 1 if x > thresh_hi
    # so we need to reduce the threshold by 1
    if cutie_style_threshs:
        thresh_hi = thresh_hi - 1



    return Thresholds(thresh_lo, thresh_hi)

def node_is_in(n, types):
    n_is_inst = lambda x: isinstance(n.module, x)
    return any(map(n_is_inst, types))

def get_node_sequences(nodes):
    # returns a list of sequences of nodes like l from a list of nodes
    # representing a quantized net (in order):
    # either:
    # l[0] -> linear (conv/FC) layer
    # l[1] -> bn layer
    # l[2] -> act layer
    # or:

    # l[0] -> linear (conv/FC) layer
    # l[1] -> pool layer/bn_layer
    # l[2] -> bn layer/pool_layer
    # l[3] -> act layer

    out = []
    for i, n in enumerate(nodes):
        try:
            if i < len(nodes)-1:
                if node_is_in(n, linear_types):
                    if node_is_in(nodes[i+1], bn_types) and node_is_in(nodes[i+2], act_types):
                        out.append(nodes[i:i+3])
                    elif node_is_in(nodes[i+1], pool_types+bn_types) and node_is_in(nodes[i+2], bn_types+pool_types) and node_is_in(nodes[i+3], act_types):
                        out.append(nodes[i:i+4])
            else:
                # we assume that the last layer of the network is a linear classifier
                if node_is_in(n, linear_types):
                    out.append([nodes[i]])
        except:
            break
    return out

def convert_net(net, in_size : torch.Tensor, dbg=False, cutie_style_threshs=False):
    # takes a quantized TNN using HTanH activations (must be left in the net)
    net.cpu()
    net_nodes = lwg.build_nodes_list(net)
    node_sequences = get_node_sequences(net_nodes)
    size = torch.tensor(in_size)
    size = size[-2:]
    prev_s = None
    for l_idx, s in enumerate(node_sequences):
        conv_node = s[0]
        n_d = int(conv_node.module.__class__.__name__[-2]) # should be 1 or 2
        assert n_d in [1,2], "Unexpected class - expected Conv1d/Conv2d, got {}".format(conv_node.module.__class__.__name__)
        # we need to provide an adaptive_kernel argument to get_threshs.. this is kinda ugly
        adaptive_kernel = None
        size = torch.div(size, torch.tensor(conv_node.module.stride), rounding_mode='trunc')

        # This processes the last linear classification module and returns the final network and argmax offsets
        if l_idx == len(node_sequences)-1:
            # we assume that the last layer of the network is a linear classifier
            # in that case, just convert the weights to ternary, from {-eps_w,
            # 0, eps_w} to {-1, 0, 1}
            if isinstance(conv_node.module, (_PACTLinOp)):
                eps_w = conv_node.module.get_eps_w()
                weights_to_sum = conv_node.module.weight_int
            else:
                eps_w = torch.ones([])
                weights_to_sum = conv_node.module.weight_frozen
            if isinstance(prev_s[-1].module, _PACTActivation):
                prev_activation_eps = prev_s[-1].module.get_eps()
            else:
                prev_activation_eps = torch.ones([])

            if conv_node.module.bias is not None:
                bias = conv_node.module.bias.data
            else:
                bias = torch.zeros(conv_node.module.out_channels)


            weights_sum = weights_to_sum.sum(dim=(1,2))
            argmax_offsets = bias/(prev_activation_eps*eps_w.reshape((-1)))
            if isinstance(prev_s[-1].module, PACTUnsignedAct):
                argmax_offsets += weights_sum
                if conv_node.module.padding_mode in ['eps', 'ones']:
                    # if we used eps padding during training, the equivalent is
                    # using zeros. Probably, the classifier won't b padding
                    # anything anyway but just to be safe
                    conv_node.module.padding_mode = 'zeros'
                else:
                        # if we did zero padding, we use -1 to pad
                    conv_node.module.padding_mode = 'neg_ones'
            if isinstance(conv_node.module, _PACTLinOp):
                conv_node.module.weight = torch.nn.Parameter(conv_node.module.weight_int)
                conv_node.module.clip_lo = torch.nn.Parameter(conv_node.module.clip_lo / eps_w)
                conv_node.module.clip_hi = torch.nn.Parameter(conv_node.module.clip_hi / eps_w)
            elif isinstance(conv_node.module, (INQConv1d, INQConv2d, INQLinear)):
                conv_node.module.weight = torch.nn.Parameter(conv_node.module.weight_frozen)
            else:
                assert False, f"Classifier of unknown class {type(conv_node.module)}"


            conv_node.module.bias = None


            # .. and we are done - move back to GPU!
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net.to(device=dev)
            return net, argmax_offsets, eps_w

        if len(s) == 4:
            # if the sequence length is 4, there is a pooling node in the
            # sequence. It's either element 1 (pool_bn layer order) or element
            # 2 (bn_pool layer order)
            pool_node, bn_node = (s[1], s[2]) if node_is_in(s[1], pool_types) else (s[2], s[1])
            if node_is_in(pool_node, adaptive_pool_types):
                adaptive_pool = True
                adaptive_kernel = size/torch.tensor(pool_node.module.output_size)
                adaptive_kernel = tuple(el.item() for el in adaptive_kernel.int())
                size = torch.tensor(pool_node.module.output_size)
                pool_node.kernel_size = adaptive_kernel
            else:
                adaptive_pool = False
                size = torch.div(size, torch.tensor(pool_node.module.stride), rounding_mode='trunc')
            i = 1

            if node_is_in(pool_node, avgpool_types):
                avgpool = True
            else:
                avgpool = False
        else:
            assert len(s) == 3, "get_node_sequences returned a bad sequence of length {}".format(len(s))
            i = 0
            avgpool = False
            bn_node = s[1]

        # we need to catch negative BN "weights"!
        # for channels where the BN gamma is negative, the weights need to be
        # flipped
        if bn_node.module.affine:
            flip_weights = bn_node.module.weight.data < 0 # flip the weights where gamma is negative
            with torch.no_grad():
                conv_node.module.weight[flip_weights, ...] *= -1

        # replace nodes:
        # - conv stays the same
        # - avg pool becomes conv2d with all-1 kernel
        if avgpool:
            n_chans = conv_node.module.out_channels
            if not adaptive_pool:
                pool_k = pool_node.module.kernel_size
            else:
                pool_k = adaptive_kernel
            if isinstance(pool_node.module, AvgPool1d):
                conv_cls = Conv1d
            elif isinstance(pool_node.module, AvgPool2d) or isinstance(pool_node.module, AdaptiveAvgPool2d):
                conv_cls = Conv2d
            else:
                assert False, "Weird: avgpool is True but pool_node is neither avgpool1d nor avgpool2d..."
            pool_conv = conv_cls(n_chans, n_chans, kernel_size=pool_k, stride=pool_k, groups=n_chans, bias=False)
            pool_conv.weight.data = torch.ones_like(pool_conv.weight.data)
            pool_conv.weight.needs_grad = False
            pool_conv.is_avgpool = True

            lwr.replace_module(net, pool_node.path, pool_conv)

        # - max pool stays the same
        # - BN becomes identity
        lwr.replace_module(net, bn_node.path, Identity())
        # - Replace activation with the correct thresholds
        act_node = s[i+2]
        threshs = get_threshs(s, prev_s, adaptive_kernel, cutie_style_threshs, first_layer=(l_idx==0))
        tern_act = TernaryActivation(threshs.lo, threshs.hi, n_d=n_d, dbg=dbg, cutie_style_threshs=cutie_style_threshs)
        lwr.replace_module(net, act_node.path, tern_act)
        # - Convolution weights are converted from {-eps_w, 0, eps_w} to {-1,
        # - 0, 1}
        if isinstance(conv_node.module, _PACTLinOp):
            eps_w = conv_node.module.get_eps_w()
            conv_node.module.weight = torch.nn.Parameter(conv_node.module.weight_int)
            conv_node.module.clip_lo = torch.nn.Parameter(conv_node.module.clip_lo / eps_w)
            conv_node.module.clip_hi = torch.nn.Parameter(conv_node.module.clip_hi / eps_w)
        elif isinstance(conv_node.module, (INQConv1d, INQConv2d, INQCausalConv1d, INQLinear)):
            conv_node.module.weight = torch.nn.Parameter(conv_node.module.weight_frozen)
        # - Convolution padding_mode is changed and bias is set to False
        # first conv still pads with 0, the following convs pad with -1 if
        # using unsigned activations
        # 'zeros' enables padding with 0, 'neg_ones' enables padding with -1.
        # when using eps padding, we don't need to do this.
        if l_idx>0 and isinstance(prev_s[-1].module, PACTUnsignedAct)  and conv_node.module.padding_mode not in ['eps', 'ones']:
            conv_node.module.padding_mode = 'neg_ones'
            print("===========WARNING!!!===========")
            print(f"Layer {conv_node.name}'s padding mode is set to {conv_node.module.padding_mode} but we are using unsigned activations - CUTIE can not handle anything but zero-padding so the mapped network will have bad accuracy! Use the 'eps' (for non-unit quantization step sizes) or 'ones' (with unit quantization step size) or  padding modes to avoid this! Setting padding mode to 'neg_ones' so the resulting network is equivalent to the fake-quantized one.")
        else:
            conv_node.module.padding_mode = 'zeros'
        conv_node.module.bias = None # note: not sure if this the correct way to do
        prev_s = s

def export_data(data, out_fn, mode, split=1,stride=0):
    split_dim = 1 if len(data.shape) == 4 else 0
    if mode == 'complete':
        # export the whole tensor as is
        np.save(f'{out_fn}', data.squeeze().numpy())
    elif mode == 'split':
        d_split = torch.split(data, split, dim=split_dim)
        for i, d in enumerate(d_split):
            np.save(f'{out_fn}_{i}', d.squeeze().numpy())
    # save only unique frames (need to provide)
    elif mode == 'cat':
        d_split = torch.split(data, split, dim=split_dim)
        inp = d_split[0]
        for i in d_split[1:]:
            if split_dim:
                inp = torch.cat((inp, i[:, -stride:,...]), dim=split_dim)
            else:
                inp = torch.cat((inp, i[-stride:,...]), dim=split_dim)
        np.save(f'{out_fn}', inp.squeeze().numpy())



def export_net(net, export_path, data, name, subnet=None, split_input=None):
    Path(export_path).mkdir(parents=True, exist_ok=True)
    if subnet is not None:
        exp_net = getattr(net, subnet)
    else:
        exp_net = net

    node_list = lwg.build_nodes_list(exp_net)
    module_list = [l.module for l in node_list]
    layer_idx = 0

    def make_file_name(kind):
        n = name
        if all(k not in kind for k in ('input', 'output')):
            n += f'_l{layer_idx}'
        n += f'_{kind}'
        if kind == 'config':
            n += '.json'

        n = os.path.join(export_path, n)
        return n


    if split_input is not None:
        export_data(data, make_file_name('input'), 'split', split_input)
    else:
        export_data(data, make_file_name('input'), 'complete')

    def export_conv_layer(m):
        wt_name = make_file_name('weights')
        if isinstance(m, (INQConv1d, INQCausalConv1d, INQConv2d)):
            np.save(wt_name, m.weight_frozen.numpy())
        else:
            np.save(wt_name, m.weight_int.numpy())

    def export_threshs(t):
        thresh_name = make_file_name('thresh_lo')
        threshs = t.thresh_lo.squeeze().numpy()
        np.save(thresh_name, threshs)
        thresh_name = make_file_name('thresh_hi')
        threshs = t.thresh_hi.squeeze().numpy()
        np.save(thresh_name, threshs)

    def export_layer(conv_module, act_module, l_dict):
        export_conv_layer(conv_module)
        if act_module is not None:
            export_threshs(act_module)
        config_name = make_file_name('config')
        with open(config_name, 'w') as fp:
            json.dump(l_dict, fp, indent=4)

    def unpack_tuple(t):
        if len(t) == 1:
            return t[0]
        return t

    layer_dict = {}
    for m in module_list:
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            try:
                is_avgpool = m.is_avgpool
            except AttributeError:
                is_avgpool = False
            if is_avgpool:
                k = expand_kernel(m)
                if k in ((1,1), (1,)):
                    # this is a "pool_conv" layer which we throw away because it
                    # does not do anything
                    continue
                layer_dict['pool_type'] = 'avg'
                layer_dict['pool_k'] = list(k)
                layer_dict['pooling'] = True
            else:
                layer_dict['conv_type'] = m.__class__.__name__[-2:]
                layer_dict['conv_stride'] = unpack_tuple(m.stride) # this can bite us...
                layer_dict['pooling'] = False
                layer_dict['conv_in_ch'] = m.in_channels
                layer_dict['conv_out_ch'] = m.out_channels
                layer_dict['fp_out'] = True
                layer_dict['conv_k'] = unpack_tuple(m.kernel_size)
                if not isinstance(m, INQCausalConv1d):
                    layer_dict['conv_padding'] = unpack_tuple(m.padding) # this can bite us...
                else:
                    layer_dict['conv_padding'] = "causal"

                if isinstance(m, nn.Conv1d):
                    layer_dict['dilation'] = unpack_tuple(m.dilation)
                    layer_dict['n_tcn_steps'] = exp_net.sequence_length

                conv_module = m

        elif isinstance(m, tuple(maxpool_types)):
            k = expand_kernel(m)
            layer_dict['pooling'] = True
            layer_dict['pool_type'] = 'max'
            layer_dict['pool_k'] = list(k)

        elif isinstance(m, TernaryActivation):
            layer_dict['fp_out'] = False
            export_layer(conv_module, m, layer_dict)
            layer_dict.clear()
            layer_idx += 1

    if conv_module is module_list[-1]:
        # export the last layer
        export_layer(conv_module, None, layer_dict)

    with torch.no_grad():
        output = net(data.unsqueeze(0))
    export_data(output, make_file_name('output'), 'complete')
