#
# pact_export.py
#
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
#
# Copyright (c) 2023 ETH Zurich.
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
import numpy as np
import json

import torch
from torch.onnx.utils import _model_to_graph as mtg
from torch import nn
import torchvision

import onnx
from onnx import shape_inference

import quantlib.editing.fx as qlfx
from quantlib.editing.fx.util import module_of_node
from quantlib.editing.lightweight import LightweightGraph
from quantlib.algorithms.pact import RequantShift, PACTIntegerLayerNorm, PACTIntegerGELU, PACTWrapMHSA, PACTWrapModule


def export_net(net: nn.Module,
               name: str,
               out_dir: str,
               eps_in: float,
               in_data: torch.Tensor,
               integerize: bool = True,
               n_levels_in=256,
               D: float = 2**24,
               opset_version: int = 10,
               code_size=0):
    net = net.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = f"{name}.onnx"
    onnx_path = out_path.joinpath(onnx_file)

    if integerize:
        net_traced = qlfx.passes.pact.PACT_symbolic_trace(net)

        int_pass = qlfx.passes.pact.IntegerizePACTNetPass(
            shape_in=in_data.shape,
            eps_in=eps_in,
            D=D,
            n_levels_in=n_levels_in,
            requant_node=True)
        net_integerized = int_pass(net_traced)
    else:
        net_integerized = net

    # First export an ONNX graph without shape inference
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
        "_retain_param_name": True
    }
    try:
        torch.onnx._export(net_integerized.to('cpu'), (in_data, ),
                            str(onnx_path),
                            opset_version=opset_version,
                            custom_opsets={"PACTOps": 1},
                            onnx_shape_inference=False,
                            **kwargs)
        graph, _, _ = mtg(net_integerized.to('cpu'), in_data, **kwargs)

    except torch.onnx.CheckerError:
        print("Disregarding PyTorch ONNX CheckerError...")

    # Infer the shapes on the ONNX graph
    varDict = {}
    for i in graph.nodes():
        varDict[i.output().debugName()] = i.output().type().sizes()

    onnxModel = onnx.load_model(str(onnx_path))
    for key,value in varDict.items():
        onnxModel.graph.value_info.append(onnx.helper.make_tensor_value_info(key, onnx.TensorProto.FLOAT, value))

    onnx.save_model(onnxModel, str(onnx_path))

    # Pass a test input through the model and log the intermediate activations

    # Make a forward hook to dump outputs of RequantShift layers
    integerized_nodes = LightweightGraph.build_nodes_list(
        net_integerized, leaf_types=(PACTWrapMHSA, ))
    acts = []

    def dump_hook(self, inp, outp, name):
        acts.append((name, torch.floor(outp)))

    for n in integerized_nodes:
        if isinstance(
                n.module,
            (RequantShift, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
             nn.AdaptiveAvgPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
             nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.Linear,
             PACTIntegerLayerNorm, PACTIntegerGELU, PACTWrapMHSA)):
            hook = partial(dump_hook, name=n.name)
            n.module.register_forward_hook(hook)

    # Open the supplied input image
    if in_data is not None:
        input = in_data.clone().to(dtype=torch.float64)
        net_integerized = net_integerized.to(dtype=torch.float64)
        output = net_integerized(input).to(dtype=torch.float64)

        input_np = torch.round(input.detach()).numpy()
        output_np = torch.round(output.detach()).numpy()

        np.savez(out_path.joinpath("inputs.npz"),
                 input=input_np.astype(np.int64))
        np.savez(out_path.joinpath("outputs.npz"),
                 output=output_np.astype(np.int64))

        acts_np = {}
        for _, (lname, t) in enumerate(acts):
            acts_np[lname] = torch.round(t.detach()).numpy().astype(np.int64)

        np.savez(out_path.joinpath("activations.npz"), **acts_np)

        def save_beautiful_text(t : torch.Tensor, layer_name : str, filename : str):
            t = t.squeeze(0)

            filepath = out_path.joinpath(f"{filename}.txt")
            with open(str(filepath), 'w') as fp:
                fp.write(f"# {layer_name} (shape {list(t.shape)}),\n")
                for el in t.flatten():
                    fp.write(f"{int(el)},\n")
                    
        save_beautiful_text(input_np, "input", "input")
        save_beautiful_text(output_np, "output", "output")

        print("Done")
