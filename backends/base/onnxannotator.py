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
