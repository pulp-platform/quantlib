import os
import torch.nn as nn
import onnx
from typing import Union


class ONNXAnnotator:

    def __init__(self, backendname: str):
        super(ONNXAnnotator, self).__init__()
        self._backendname = backendname

    def _annotate(self,
                  network:   nn.Module,
                  onnxproto: onnx.ModelProto) -> None:
        """This method has a side-effect: annotating ``onnx.ModelProto``."""
        raise NotImplementedError

    def annotate(self,
                 network:      nn.Module,
                 onnxfilepath: Union[os.PathLike, str]) -> None:

        # canonicalise input
        if not isinstance(onnxfilepath, str):
            onnxfilepath = str(onnxfilepath)

        # load ONNX
        onnxproto = onnx.load(onnxfilepath)

        # annotate ONNX
        self._annotate(network, onnxproto)

        # save backend-specific ONNX
        onnxfilepath = onnxfilepath.rsplit('.', 1)[0].rstrip('_NOANNOTATION') + '_' + self._backendname + '.onnx'  # TODO: generate the backend-annotated filename more elegantly
        onnx.save(onnxproto, onnxfilepath)
