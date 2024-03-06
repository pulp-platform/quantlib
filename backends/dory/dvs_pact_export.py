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
from quantlib.editing.fx.passes.pact import PACT_symbolic_trace
from quantlib.editing.lightweight import LightweightGraph
from quantlib.algorithms.pact import RequantShift, PACTIntegerLayerNorm, PACTIntegerGELU, PACTWrapMHSA, PACTWrapModule, ChannelwiseThreshold
from .dory_passes import AvgPoolWrap, DORYAdder, DORYHarmonizePass

from .pact_export import get_input_channels, annotate_onnx

def export_dvsnet(net_cnn : nn.Module, net_tcn : nn.Module, name : str, out_dir : str, eps_in : float, in_data : torch.Tensor, integerize : bool = True, D : float = 2**24, opset_version : int  = 10, change_n_levels : int = None, code_size : int = 310000, compressed : bool = False):
    net_cnn = PACT_symbolic_trace(net_cnn)
    net_tcn = PACT_symbolic_trace(net_tcn)
    #if isinstance(net_cnn, fx.GraphModule):
    cnn_window = get_input_channels(net_cnn)
    #else:
        #cnn_window = net_cnn.adapter.in_channels

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
            if (not compressed) or (module.n_levels != 3):
                n_bits = int(np.log2(module.n_levels+1.2))
                prec_dict_cnn[lname] = n_bits
            else:
                prec_dict_cnn[lname] = 1.6

    prec_dict_tcn = {}
    for lname, module in int_net_tcn.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if (not compressed) or (module.n_levels != 3):
                n_bits = int(np.log2(module.n_levels+1.2))
                prec_dict_tcn[lname] = n_bits
            else:
                prec_dict_tcn[lname] = 1.6

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

    for n, m in int_net_cnn.named_modules():
        if isinstance(m, (RequantShift, nn.MaxPool1d, nn.MaxPool2d, ChannelwiseThreshold)):
            hook = partial(dump_hook, lname=n)
            m.register_forward_hook(hook)
    for n, m in int_net_tcn.named_modules():
        if isinstance(m, (RequantShift, nn.MaxPool1d, nn.MaxPool2d, nn.Linear, ChannelwiseThreshold)):
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
                       "code reserved space": code_size,
                       "n_inputs": tcn_window,
                       "input_bits": 1.6 if compressed else 2,
                       "input_signed": True,
                       "input_shape": list(shape_in_cnn[-3:]),
                       "output_shape": list(cnn_outs[0].shape[-2:])}

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
    acts = []

    output = int_net_tcn(tcn_input.to(dtype=torch.float64))

    save_beautiful_text(tcn_input, "input", "input", out_path_tcn)
    save_beautiful_text(output, "output", "output", out_path_tcn)
    for jdx, (lname, t) in enumerate(acts):
        save_beautiful_text(t, lname, f"out_layer{jdx}", out_path_tcn)


    int_net_tcn = int_net_tcn.to(torch.float64)

    print(f"tcn input shape: {tcn_input.shape}")
    tcn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path_tcn.resolve()),
                       "code reserved space": code_size,
                       "n_inputs": 1,
                       "input_bits": 1.6 if compressed else 2,
                       "input_signed": False,
                       "input_shape": list(tcn_input.shape[-2:]),
                       "output_shape": output.shape[-1]}

    with open(out_path_tcn.joinpath(f"config_{name}_tcn.json"), "w") as fp:
        json.dump(tcn_dory_config, fp, indent=4)
    # now we pass a test input through the model and log the intermediate
    # activations
