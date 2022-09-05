#
# pact_export.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

from functools import partial
from itertools import chain 
from pathlib import Path
import json


import torch

from torch import nn, fx
import torchvision

import onnx
import numpy as np



import quantlib.editing.fx as qlfx
from quantlib.editing.fx.util import module_of_node
from quantlib.editing.lightweight import LightweightGraph
from quantlib.algorithms.pact import RequantShift
from .dory_passes import AvgPoolWrap, DORYAdder, DORYHarmonizePass

def get_input_channels(net : fx.GraphModule):
    for node in net.graph.nodes:
        if node.op == 'call_module' and isinstance(module_of_node(net, node), (nn.Conv1d, nn.Conv2d)):
            conv = module_of_node(net, node)
            return conv.in_channels

# annotate:
#   conv, FC nodes:
#     - 'weight_bits'
#     - 'bias_bits'
#   clip nodes (these represent [re]quantization nodes):
#     - 'out_bits'
#   Mul nodes:
#     - 'mult_bits' - this describes the multiplicative factor as well as the
#                    result's precision
#   Add nodes:
#     - 'add_bits' - this describes the added number's as well as the result's precision



def get_attr_by_name(node, name):
    try:
        a = [attr for attr in node.attribute if attr.name == name][0]
    except IndexError:
        a = "asdf"
    return a


def annotate_onnx(m, prec_dict : dict, requant_bits : int = 32):

# annotate all clip nodes - use the clip limits to determine the number of
# bits; in a properly exported model this would be done based on
# meta-information contained in the pytorch graph.
    clip_nodes = [n for n in m.graph.node if n.op_type == "Clip"]
    for i, n in enumerate(clip_nodes):
        lower = get_attr_by_name(n, "min").f
        #assert lower == 0.0, "clip node {} has lower clip bound {} not equal to zero!".format(n.name, lower)
        upper = get_attr_by_name(n, "max").f
        n_bits = int(np.round(np.log2(upper-lower+1.0)))
        n_bits = n_bits if n_bits <= 8 else 32
        precision_attr = onnx.helper.make_attribute(key='out_bits', value=n_bits)
        n.attribute.append(precision_attr)


    conv_fc_nodes = [n for n in m.graph.node if n.op_type in ['Conv', 'Gemm', 'MatMul']]
    for n in conv_fc_nodes:
        if n.op_type == 'MatMul':
            import ipdb; ipdb.set_trace()
        weight_name = n.input[1].rstrip('.weight')
        weight_bits = prec_dict[weight_name]
        weight_attr = onnx.helper.make_attribute(key='weight_bits', value=weight_bits)
        n.attribute.append(weight_attr)
        # bias accuracy hardcoded to 32b.
        bias_attr = onnx.helper.make_attribute(key='bias_bits', value=32)
        n.attribute.append(bias_attr)

    # assume that add nodes have 32b precision?? not specified in the name...
    add_nodes_requant = [n for n in m.graph.node if n.op_type == 'Add' and not all(i.isnumeric() for i in n.input)]
    # the requantization add nodes are the ones not adding two operation nodes'
    # outputs together
    for n in add_nodes_requant:
        add_attr = onnx.helper.make_attribute(key='add_bits', value=requant_bits)
        n.attribute.append(add_attr)

    add_nodes_residual = [n for n in m.graph.node if n.op_type == 'Add' and n not in add_nodes_requant]
    for n in add_nodes_residual:
        # assume that all residual additions are executed in 8b
        add_attr = onnx.helper.make_attribute(key='add_bits', value=8)
        n.attribute.append(add_attr)

    mult_nodes = [n for n in m.graph.node if n.op_type == 'Mul']
    for n in mult_nodes:
        mult_attr = onnx.helper.make_attribute(key='mult_bits', value=requant_bits)
        n.attribute.append(mult_attr)

def export_net(net : nn.Module, name : str, out_dir : str, eps_in : float, in_data : torch.Tensor, integerize : bool = True, D : float = 2**24, opset_version : int  = 10, align_avg_pool : bool = False):
    net = net.eval()


    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = f"{name}_ql_integerized.onnx"
    onnx_path = out_path.joinpath(onnx_file)

    shape_in = in_data.shape

    if integerize:
        net_traced = qlfx.passes.pact.PACT_symbolic_trace(net)

        int_pass = qlfx.passes.pact.IntegerizePACTNetPass(shape_in=shape_in,  eps_in=eps_in, D=D)
        net_integerized = int_pass(net_traced)
    else: # assume the net is already integerized
        net_integerized = net

    if align_avg_pool:
        align_avgpool_pass = DORYHarmonizePass(in_shape=shape_in)
        net_integerized = align_avgpool_pass(net_integerized)

    integerized_nodes = LightweightGraph.build_nodes_list(net_integerized, leaf_types=(AvgPoolWrap, DORYAdder))

    # the integerization pass annotates the conv layers with the number of
    # weight levels. from this information we can make a dictionary of the number of
    # weight bits.
    prec_dict = {}

    for lname, module in net_integerized.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            n_bits = int(np.log2(module.n_levels+1.2))
            prec_dict[lname] = n_bits

    #first export an unannotated ONNX graph
    test_input = torch.rand(shape_in)
    torch.onnx.export(net_integerized.to('cpu'),
                      test_input,
                      str(onnx_path),
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      enable_onnx_checker=False)

    #load the exported model and annotate it
    onnx_model = onnx.load(str(onnx_path))
    annotate_onnx(onnx_model, prec_dict)
    # finally, save the annotated ONNX model
    onnx.save(onnx_model, str(onnx_path))
    # now we pass a test input through the model and log the intermediate
    # activations



    # make a forward hook to dump outputs of RequantShift layers
    acts = []
    def dump_hook(self, inp, outp, name):
        # DORY wants HWC tensors
        acts.append((name, torch.floor(outp[0])))

    for n in integerized_nodes:
        if isinstance(n.module, (RequantShift, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.Linear, AvgPoolWrap, DORYAdder)):
            hook = partial(dump_hook, name=n.name)
            n.module.register_forward_hook(hook)

    # open the supplied input image
    if in_data is not None:
        im_tensor = in_data.clone()
        net_integerized = net_integerized.to(dtype=torch.float64)
        output = net_integerized(im_tensor.to(dtype=torch.float64))

        # now, save everything into beautiful text files
        def save_beautiful_text(t : torch.Tensor, layer_name : str, filename : str):
            t = t.squeeze(0)
            if t.dim()==3:
                # expect a (C, H, W) tensor - DORY expects (H, W, C)
                t = t.permute(1,2,0)
            elif t.dim()==2:
                # expect a (C, D) tensor - DORY expects (D, C)
                t = t.permute(1,0)
            else:
                print(f"Not permuting output of layer {layer_name}...")

            filepath = out_path.joinpath(f"{filename}.txt")
            with open(str(filepath), 'w') as fp:
                fp.write(f"# {layer_name} (shape {list(t.shape)}),\n")
                for el in t.flatten():
                    fp.write(f"{int(el)},\n")
        save_beautiful_text(im_tensor, "input", "input")
        save_beautiful_text(output, "output", "output")
        for i, (lname, t) in enumerate(acts):
            save_beautiful_text(t, lname, f"out_layer{i}")

        
    cnn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path.resolve()),
                       "code reserved space": 195000,
                       "n_inputs": 1,
                       "input_bits": 8,
                       "input_signed": True}

    with open(out_path.joinpath(f"config_{name}_cnn.json"), "w") as fp:
        json.dump(cnn_dory_config, fp, indent=4)

    #done!


def export_dvsnet(net_cnn : nn.Module, net_tcn : nn.Module, name : str, out_dir : str, eps_in : float, in_data : torch.Tensor, integerize : bool = True, D : float = 2**24, opset_version : int  = 10, change_n_levels : int = None):
    if isinstance(net_cnn, fx.GraphModule):
        cnn_window = get_input_channels(net_cnn)
    else:
        cnn_window = net_cnn.adapter.in_channels

    net_cnn = net_cnn.eval()
    net_tcn = net_tcn.eval()
    if change_n_levels:
        for m in chain(net_cnn.modules(), net_tcn.modules()):
            if isinstance(m, RequantShift):
                m.n_levels_out.data = torch.Tensor([change_n_levels])
    out_path_cnn = Path(out_dir).joinpath('cnn')
    out_path_tcn = Path(out_dir).joinpath('tcn')
    out_path_cnn.mkdir(parents=True, exist_ok=True)
    out_path_tcn.mkdir(parents=True, exist_ok=True)
    onnx_file_cnn = f"{name}_cnn_ql_integerized.onnx"
    onnx_file_tcn = f"{name}_tcn_ql_integerized.onnx"
    onnx_path_cnn = out_path_cnn.joinpath(onnx_file_cnn)
    onnx_path_tcn = out_path_tcn.joinpath(onnx_file_tcn)

    cnn_wins = torch.split(in_data, cnn_window, dim=1)
    tcn_window = len(cnn_wins)
    shape_in_cnn = cnn_wins[0].shape
    # atm no integerization is done here. Assume the nets are already integerized
    int_net_cnn = net_cnn
    int_net_tcn = net_tcn

    # the integerization pass annotates the conv layers with the number of
    # weight levels. from this information we can make a dictionary of the number of
    # weight bits.
    prec_dict_cnn = {}
    for lname, module in int_net_cnn.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            n_bits = int(np.log2(module.n_levels+1.2))
            prec_dict_cnn[lname] = n_bits
    prec_dict_tcn = {}
    for lname, module in int_net_tcn.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            n_bits = int(np.log2(module.n_levels+1.2))
            prec_dict_tcn[lname] = n_bits

    #first export an unannotated ONNX graph
    test_input_cnn = torch.rand(shape_in_cnn)
    torch.onnx.export(int_net_cnn.to('cpu'),
                      test_input_cnn,
                      str(onnx_path_cnn),
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True)

    int_net_cnn = int_net_cnn.to(torch.float64)
    #load the exported model and annotate it
    onnx_model_cnn = onnx.load(str(onnx_path_cnn))
    annotate_onnx(onnx_model_cnn, prec_dict_cnn)
    # finally, save the annotated ONNX model
    onnx.save(onnx_model_cnn, str(onnx_path_cnn))
    # now we pass a test input through the model and log the intermediate
    # activations

    # make a forward hook to dump outputs of RequantShift layers
    acts = []
    def dump_hook(self, inp, outp, lname):
        # DORY wants HWC tensors
        acts.append((lname, outp[0]))

    int_acts = []
    def dump_hook_dbg(self, inp, outp, lname):
        # DORY wants HWC tensors
        int_acts.append((lname, outp[0]))

    for n, m in int_net_cnn.named_modules():
        if isinstance(m, (RequantShift, nn.MaxPool1d, nn.MaxPool2d)):
            hook = partial(dump_hook, lname=n)
            m.register_forward_hook(hook)
    for n, m in int_net_cnn.named_modules():
        if isinstance(m, (nn.Conv2d)):
            hook = partial(dump_hook_dbg, lname=n)
            m.register_forward_hook(hook)
    for n, m in int_net_tcn.named_modules():
        if isinstance(m, (RequantShift, nn.MaxPool1d, nn.MaxPool2d, nn.Linear)):
            hook = partial(dump_hook, lname=n)
            m.register_forward_hook(hook)

    # save everything into beautiful text files
    def save_beautiful_text(t : torch.Tensor, layer_name : str, filename : str, out_path : Path):
        t = t.squeeze(0)
        if t.dim()==3:
            # expect a (C, H, W) tensor - DORY expects (H, W, C)
            t = t.permute(1,2,0)
        elif t.dim()==2:
            # expect a (C, D) tensor - DORY expects (D, C)
            t = t.permute(1,0)
        else:
            print(f"Not permuting output of layer {layer_name}...")

        filepath = out_path.joinpath(f"{filename}.txt")
        np.savetxt(str(filepath), t.detach().flatten().numpy().astype(np.int32), delimiter=',', header=f"# {layer_name} (shape {list(t.shape)}),", fmt="%1d,")
        #with open(str(filepath), 'w') as fp:
        #    fp.write(f"# {layer_name} (shape {list(t.shape)}),\n")
        #    for el in t.flatten():p
        #        fp.write(f"{int(el)},\n")

    # save the whole input tensor
    save_beautiful_text(in_data, "input", "input", out_path_cnn)
    cnn_outs = []
    # feed the windows one by one to the cnn
    for idx, cnn_win in enumerate(cnn_wins):
        cnn_win_out = int_net_cnn(cnn_win.to(dtype=torch.float64))
        cnn_outs.append(cnn_win_out)
        save_beautiful_text(cnn_win, f"input_{idx}", f"input_{idx}", out_path_cnn)
        save_beautiful_text(cnn_win_out, f"output_{idx}", f"output_{idx}", out_path_cnn)
        for jdx, (lname, t) in enumerate(acts):
            save_beautiful_text(t, lname, f"out_{idx}_layer{jdx}", out_path_cnn)
        acts = []
    cnn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path_cnn.resolve()),
                       "code reserved space": 310000,
                       "n_inputs": tcn_window,
                       "input_bits": 2,
                       "input_signed": True}
    with open(out_path_cnn.joinpath(f"config_{name}_cnn.json"), "w") as fp:
        json.dump(cnn_dory_config, fp, indent=4)

    #first export an unannotated ONNX graph
    tcn_input = torch.stack(cnn_outs, dim=2)
    shape_in_tcn = tcn_input.shape
    test_input_tcn = torch.rand(shape_in_tcn)
    torch.onnx.export(int_net_tcn.to('cpu'),
                      test_input_tcn,
                      str(onnx_path_tcn),
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True)

    #load the exported model and annotate it
    onnx_model_tcn = onnx.load(str(onnx_path_tcn))
    annotate_onnx(onnx_model_tcn, prec_dict_tcn)
    # finally, save the annotated ONNX model
    onnx.save(onnx_model_tcn, str(onnx_path_tcn))
    int_net_tcn = int_net_tcn.to(torch.float64)
    int_acts = []
    acts = []
    output = int_net_tcn(tcn_input.to(dtype=torch.float64))

    save_beautiful_text(tcn_input, "input", "input", out_path_tcn)
    save_beautiful_text(output, "output", "output", out_path_tcn)
    for jdx, (lname, t) in enumerate(acts):
        save_beautiful_text(t, lname, f"out_layer{jdx}", out_path_tcn)


    int_net_tcn = int_net_tcn.to(torch.float64)

    tcn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path_tcn.resolve()),
                       "code reserved space": 310000,
                       "n_inputs": 1,
                       "input_bits": 2,
                       "input_signed": False}

    with open(out_path_tcn.joinpath(f"config_{name}_tcn.json"), "w") as fp:
        json.dump(tcn_dory_config, fp, indent=4)
    # now we pass a test input through the model and log the intermediate
    # activations
