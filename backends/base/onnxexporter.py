import os
import torch
import torch.nn as nn
from typing import Optional

from .onnxannotator import ONNXAnnotator


class ONNXExporter(object):
    """Base class to export QuantLib-trained networks to ONNX models."""

    def __init__(self, annotator: ONNXAnnotator):
        super(ONNXExporter, self).__init__()
        self._annotator = annotator
        self._onnxfilepath = None

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

        # export the network (https://pytorch.org/docs/master/onnx.html#torch.onnx.export)
        torch.onnx.export(network,
                          torch.randn(input_shape),  # a dummy input to trace the `nn.Module`
                          onnxfilepath,
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=opset_version)

        # annotate the ONNX model with backend-specific information
        self._annotator.annotate(network, onnxfilepath)
