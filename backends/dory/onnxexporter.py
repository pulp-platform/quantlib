# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
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
import os
import torch
import torch.nn as nn
from typing import List, NamedTuple
from .onnxannotator import DORYAnnotator
from quantlib.backends.base import ONNXExporter
import quantlib.editing.graphs as qg
import json
from pathlib import Path
class DORYExporter(ONNXExporter):
    def __init__(self):
        annotator = DORYAnnotator()
        super(DORYExporter, self).__init__(annotator=annotator)

        
    def export_json_config(
        self,
        code_size:  int = 160000,
        nb_inputs:  int = 1,
        input_bits: int = 8,
        input_signed: bool = True,
        name: str = "DEFAULT"
    ):
        
        onnx_file = self._onnxfilepath.rsplit('.', 1)[0].rstrip('_NOANNOTATION') + '_DORY.onnx'  # TODO: generate the backend-annotated filename more elegantly
        name = self._onnxname

        cnn_dory_config = {
            "BNRelu_bits": 32,
            "onnx_file": onnx_file,
            "code reserved space": code_size,
            "n_inputs": nb_inputs, # TODO retrieve this info from QL graph
            "input_bits": input_bits, # TODO retrieve this info from QL graph
            "input_signed": input_signed # TODO retrieve this info from QL graph
        }
        jsonfilepath = Path(self._onnxfilepath).parent
        with open(jsonfilepath.joinpath(f"config_{name}.json"), "w") as fp:
            json.dump(cnn_dory_config, fp, indent=4)
    @staticmethod
    def dump_features(network: nn.Module,
                      x:       torch.Tensor,
                      path:    os.PathLike) -> None:
        """Given a network, export the features associated with a given input.
        To verify the correctness of an ONNX export, DORY requires text files
        containing the values of the features for each layer in the target
        network. The format of these text files is exemplified here:
        https://github.com/pulp-platform/dory_examples/tree/master/examples/Quantlab_examples .
        """

        class Features(NamedTuple):
            module_name: str
            features:    torch.Tensor

        def export_to_txt(module_name: str, filename: str, t: torch.Tensor):

            try:  # for the output, this step is not applicable
                t = t.squeeze().permute(1, 2, 0)  # PyTorch's `nn.Conv2d` layers output CHW arrays, but DORY expects HWC arrays
            except RuntimeError:
                pass  # I won't permute the features of this module

            filepath = os.path.join(path, f"{filename}.txt")
            with open(str(filepath), 'w') as fp:
                fp.write(f"# {module_name} (shape {list(t.shape)}),\n")
                for c in t.flatten():
                    fp.write(f"{int(c)},\n")

        # Since PyTorch uses dynamic graphs, we don't have symbolic handles
        # over the inner array. Therefore, we use PyTorch hooks to dump the
        # outputs of Requant layers.
        features: List[Features] = []

        def hook_fn(self, in_: torch.Tensor, out_: torch.Tensor, module_name: str):
            # DORY wants HWC tensors
            features.append(Features(module_name=module_name, features=out_.squeeze(0)))

        # the core dump functionality starts here

        # 1. set up hooks to intercept features
        for n, m in network.named_modules():
            if isinstance(m, qg.nn.Requantisation):  # TODO: we are tacitly assuming that these layers will always output 4D feature arrays
                hook = partial(hook_fn, module_name=n)
                m.register_forward_hook(hook)

        # 2. propagate the supplied input through the network; the hooks will capture the features
        x = x.clone()
        y = network(x.to(dtype=torch.float32))

        # 3. export input, features, and output to text files
        export_to_txt('input', 'input', x)
        for i, (module_name, f) in enumerate(features):
            export_to_txt(module_name, f"out_layer{i}", f)
        export_to_txt('output', f"out_layer{len(features)}", y)
        export_to_txt('output', 'output', y)
