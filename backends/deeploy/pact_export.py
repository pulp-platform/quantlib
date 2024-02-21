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

from packaging.version import Version
from typing import Tuple, Union
from functools import partial
from pathlib import Path
import numpy as np

import torch
from torch import nn

import onnx
import onnxruntime

import quantlib.editing.fx as qlfx
from quantlib.editing.lightweight import LightweightGraph
from quantlib.algorithms.pact import RequantShift, PACTIntegerLayerNorm, PACTIntegerGELU, PACTWrapMHSA, PACTWrapModule, PACTIntegerHardswish, PACTTrueIntegerDiv, PACTIntegerRMSNorm

# Import ONNX runtime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.transformers.optimizer import optimize_model
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

@dataclass
class OptimizationConfig:
    enable_gelu: bool = True
    enable_layer_norm: bool = True
    enable_attention: bool = True
    enable_skip_layer_norm: bool = False
    enable_embed_layer_norm: bool = False
    enable_bias_skip_layer_norm: bool = False
    enable_bias_gelu: bool = False
    enable_gelu_approximation: bool = False
    enable_qordered_matmul: bool = False
    enable_shape_inference: bool = True
    attention_mask_format: int = 3
    use_multi_head_attention: bool = False
    enable_gemm_fast_gelu: bool = False

# By default all attributes of the original node are removed and replaced
# with the default attributes of the temporary replacement node.
NODES_MAPPING = {
    # Replace nodes with 3 inputs with LayerNorm
    "iLayerNorm": {
        "op_type": "LayerNormalization",
    },
    "iRMSNorm": {
        "op_type": "LayerNormalization",
    },
    "RequantShift": {
        "op_type": "LayerNormalization",
    },
    # Replace nodes with 1 input with ReLU
    "iSoftmax": {
        "op_type": "Relu",
    },
    "ITAMax": {
        "op_type": "Relu",
    },
     "ITAPartialMax": {
        "op_type": "Relu",
    },
    "iGELU": {
        "op_type": "Relu",
    },
    "IntegerDiv": {
        "op_type": "Div",
    },
    "TrueIntegerDiv": {
        "op_type": "Relu",
    },
    "iHardswish": {
        "op_type": "Relu",
    },
    # Copy original attribute to replaced node
    "IntegerMean": {
        "op_type": "ReduceMean",
        "attr": "copy"
    },
    # # Set custom attributes in replaced node
    # "IntegerMean": {
    #     "op_type": "ReduceMean",
    #     "attr": {
    #         "axes": [1],
    #         "keepdims": 0
    #     }
    # },
}


def save_beautiful_text(t: np.ndarray, layer_name: str, filepath: str):
    with open(str(filepath), 'w') as fp:
        fp.write(f"# {layer_name} (shape {list(t.shape)}),\n")

        if t.ndim == 3:
            t = t.reshape(1, t.shape[0], t.shape[1], t.shape[2])
        elif t.ndim == 2:
            t = t.reshape(1, 1, t.shape[0], t.shape[1])
        elif t.ndim == 1:
            t = t.reshape(1, 1, 1, t.shape[0])

        # print(layer_name, t.max())
        for batch in t:
            if t.ndim >= 4: fp.write("[\n")
            for channel in batch:
                if t.ndim >= 4:
                    fp.write("  [\n  ")
                elif t.ndim >= 3:
                    fp.write(" [\n")
                for row in channel:
                    for i in row:
                        if batch.max() == 0 or np.log2(np.abs(batch.max())) <= 8:
                            fp.write(f"{int(i):4d} ")
                        elif np.log2(np.abs(batch.max())) <= 16:
                            fp.write(f"{int(i):6d} ")
                        else:
                            fp.write(f"{int(i):11d} ")

                    if t.ndim >= 4:
                        fp.write("\n  ")
                    elif t.ndim >= 3:
                        fp.write("\n")

                if t.ndim >= 3: fp.write("]\n")
            if t.ndim >= 4: fp.write("]\n")

def export_net(net: nn.Module,
               name: str,
               out_dir: str,
               eps_in: float,
               in_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
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
            requant_node=True,
            export_layernorm_node = True,
            export_softmax_node = True,
            export_gelu_node = True,
            export_div_node = True)
        net_integerized = int_pass(net_traced)
    else:
        net_integerized = net

    # First export an ONNX graph without shape inference
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
    }
    try:
        torch.onnx._export(net_integerized.to('cpu'), in_data,
                           str(onnx_path),
                           opset_version=opset_version,
                           custom_opsets={"PACTOps": 1},
                           onnx_shape_inference=False,
                           verbose=False,
                           keep_initializers_as_inputs = False,
                           **kwargs)

    except torch.onnx.CheckerError:
        print("Disregarding PyTorch ONNX CheckerError...")

    onnxModel = onnx.load_model(str(onnx_path))

    # Rename nodes
    if torch.__version__ < '1.13':
        pass
        # For pytorch < 1.13 overwrite name with name of the traced nodes
        # WIESEP: Be careful, I am not sure if the order of the nodes in the onnx model and traced net is always the same!
        # for i, node in enumerate(net_integerized.graph.nodes):
        #     if i == 0: continue  # First node is the input
        #     if i > len(onnxModel.graph.node): break  # Last node is the ouput
        #     onnxModel.graph.node[i - 1].name = node.name
    else:
        # # For pytorch >= 1.13 preserves the original scope names with some changes
        # # Replace "/" characters
        for n in onnxModel.graph.node:
            n.name = n.name.replace("/", "_")

            for i, name in enumerate(n.input):
                n.input[i] = name.replace("/", "_")

            for i, name in enumerate(n.output):
                n.output[i] = name.replace("/", "_")

        for i, info in enumerate(onnxModel.graph.value_info):
            onnxModel.graph.value_info[i].name = info.name.replace("/", "_")


    # Replace custom nodes with standard ones for optimization
    replaced_nodes = {}
    for n in onnxModel.graph.node:
        if n.op_type in NODES_MAPPING:
            replaced_nodes[n.name] = n.__deepcopy__()
            replace_node = NODES_MAPPING[n.op_type]
            n.op_type = replace_node["op_type"]
            n.domain = ""
            # Remove original attributes
            if "attr" in replace_node:
                if replace_node["attr"] == "copy":
                    pass
                else:
                    for att in n.attribute.__deepcopy__():
                        n.attribute.remove(att)
                    for key in replace_node["attr"]:
                        att = onnx.helper.make_attribute(key, replace_node["attr"][key])
                        n.attribute.append(att)
            else:
                for att in n.attribute.__deepcopy__():
                    n.attribute.remove(att)

    onnx.save_model(onnxModel, str(onnx_path))

    # Optimize ONNX model with replaced nodes
    optimization_config = OptimizationConfig(
        enable_skip_layer_norm=False,
        enable_bias_gelu=False,
    )

    if Version(onnxruntime.__version__) >= Version("1.17"):
        optimization_config.enable_rotary_embeddings = False

    optimizer = optimize_model(str(onnx_path), optimization_options=optimization_config)
    optimizer.save_model_to_file(str(onnx_path))

    # Run shape inference
    onnxModel = onnx.load_model(onnx_path)
    onnxModel = SymbolicShapeInference.infer_shapes(onnxModel)
    onnx.save_model(onnxModel, onnx_path)

    # Switch back custom nodes
    for n in onnxModel.graph.node:
        if n.name in replaced_nodes:
            n.op_type = replaced_nodes[n.name].op_type
            # Remove attributes of the standard nodes
            for att in n.attribute.__deepcopy__():
                n.attribute.remove(att)
            # Restore original attributes
            for att in replaced_nodes[n.name].attribute:
                n.attribute.append(att)

    onnx.save_model(onnxModel, str(onnx_path))

    # Pass a test input through the model and log the intermediate activations
    # Make a forward hook to dump outputs of RequantShift layers
    acts = []

    def dump_hook(self, inp, outp, name):
        _name = name.lower().replace(".", "_")
        if isinstance(inp, torch.Tensor):
            inpNan = any([torch.sum(torch.isnan(inp)) > 0])
        else:
            inpNan = any([torch.sum(torch.isnan(y)) > 0 for y in inp if isinstance(y, torch.Tensor)])
        outpNan = torch.sum(torch.isnan(outp)) > 0
        if inpNan or outpNan:
            raise Exception("Caught NaN in intermediate activations!")

        acts.append((_name, torch.round(outp)))

    integerized_nodes = LightweightGraph.build_nodes_list(net_integerized, leaf_types=(PACTWrapMHSA, ))
    for n in integerized_nodes:
        if isinstance(
                n.module,
            (RequantShift, nn.Conv2d, nn.Conv1d, nn.AdaptiveAvgPool1d,
             nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AvgPool1d,
             nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d,
             nn.MaxPool3d, nn.Linear, PACTIntegerLayerNorm, PACTIntegerGELU, PACTIntegerRMSNorm,
             PACTIntegerHardswish, PACTTrueIntegerDiv,
             PACTWrapMHSA)):
            hook = partial(dump_hook, name=n.name)
            n.module.register_forward_hook(hook)

    # Open the supplied input image
    if in_data is not None:

        net_integerized = net_integerized.to(dtype=torch.float64)

        if isinstance(in_data, torch.Tensor):
            input = in_data.clone().to(dtype=torch.float64)
            input_np = [torch.round(input.detach()).numpy().astype(np.int64)]
            _output = net_integerized(input).to(dtype=torch.float64)
        else:
            input = [t.clone().to(dtype=torch.float64) for t in in_data]
            input_np = [torch.round(t.detach()).numpy().astype(np.int64) for t in input]
            _output = net_integerized(*input)

        if isinstance(_output, torch.Tensor):
            _output.to(dtype=torch.float64)
            output = _output
            output_np = [torch.round(output.detach()).numpy().astype(np.int64)]
        else:
            output = [t.to(dtype=torch.float64) for t in _output if isinstance(t, torch.Tensor)]
            output_np = [torch.round(t.detach()).numpy().astype(np.int64) for t in output]

        inputkwargs = {}
        for idx, array in enumerate(input_np):
            inputkwargs[f"input_{idx}"] = array
        outputkwargs = {}
        for idx, array in enumerate(output_np):
            outputkwargs[f"output_{idx}"] = array

        np.savez(out_path.joinpath("inputs.npz"),**inputkwargs)
        np.savez(out_path.joinpath("outputs.npz"), **outputkwargs)

        acts_np = {}
        for _, (lname, t) in enumerate(acts):
            acts_np[lname] = torch.round(t.detach()).numpy().astype(np.int64)

        np.savez(out_path.joinpath("activations.npz"), **acts_np)

        out_path.joinpath("activations/").mkdir(parents=True, exist_ok=True)

        for idx, array in enumerate(output_np):
            save_beautiful_text(array, f"input_{idx}", out_path.joinpath(f"activations/input_{idx}.txt"))
        for idx, array in enumerate(output_np):
            save_beautiful_text(array, f"output_{idx}", out_path.joinpath(f"activations/output_{idx}.txt"))
        for jdx, lname in enumerate(acts_np):
            save_beautiful_text(acts_np[lname], lname, out_path.joinpath(f"activations/act{jdx:02d}_{lname}.txt"))
