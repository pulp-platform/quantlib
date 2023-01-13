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

import os
import torch
import torch.nn as nn
from typing import Optional
import warnings

from .onnxannotator import ONNXAnnotator


class ONNXExporter(object):
    """Base class to export QuantLib-trained networks to ONNX models."""

    def __init__(self, annotator: ONNXAnnotator):
        super(ONNXExporter, self).__init__()
        self._annotator = annotator
        self._onnxfilepath = None
        self._onnxname = None

    def export(self,
               network:       nn.Module,
               input_shape:   torch.Size,
               path:          os.PathLike,
               name:          Optional[str] = None,
               opset_version: int = 10) -> None:

        # compute the name of the ONNX file
        onnxname     = name if name is not None else network._get_name()
        onnxfilename = onnxname + '_QL_NOANNOTATION.onnx'  # TODO: should the name hint at whether the network is FP, FQ, or TQ? How can we derive this information (user-provided vs. inferred from the network)?
        onnxfilepath = os.path.join(path, onnxfilename)
        self._onnxfilepath = onnxfilepath
        self._onnxname = onnxname

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # export the network (https://pytorch.org/docs/master/onnx.html#torch.onnx.export)
            torch.onnx.export(network,
                            torch.randn(input_shape),  # a dummy input to trace the `nn.Module`
                            onnxfilepath,
                            export_params=True,
                            do_constant_folding=True,
                            opset_version=opset_version)

            # annotate the ONNX model with backend-specific information
            self._annotator.annotate(network, onnxfilepath)
