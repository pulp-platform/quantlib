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
import torch.nn as nn
import onnx
from typing import Union


class ONNXAnnotator(object):
    """Base class to annotate QuantLib-exported ONNX models."""

    def __init__(self, backend_name: str):
        super(ONNXAnnotator, self).__init__()
        self._backend_name = backend_name

    def _annotate(self,
                  network:   nn.Module,
                  onnxproto: onnx.ModelProto) -> None:
        """Annotate ``onnxproto`` with backend-specific information.

        The backend-specific information might be computed from the given
        ``network`` argument.

        This method operates by side-effect on ``onnxproto``.
        """
        raise NotImplementedError

    def annotate(self,
                 network:      nn.Module,
                 onnxfilepath: Union[os.PathLike, str]) -> None:
        """A wrapper method around ``_annotate``, performing input validation
        and storage transactions.
        """

        # canonicalise input
        if not isinstance(onnxfilepath, str):
            onnxfilepath = str(onnxfilepath)

        # load ONNX
        onnxproto = onnx.load(onnxfilepath)

        # annotate ONNX
        self._annotate(network, onnxproto)

        # save backend-specific ONNX
        onnxfilepath = onnxfilepath.rsplit('.', 1)[0].rstrip('_NOANNOTATION') + '_' + self._backend_name + '.onnx'  # TODO: generate the backend-annotated filename more elegantly
        onnx.save(onnxproto, onnxfilepath)
