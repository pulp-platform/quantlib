from functools import partial
import os
import torch
import torch.nn as nn
from typing import List, NamedTuple

from .onnxannotator import DORYAnnotator
import quantlib.newediting.editing as qle
import quantlib.newediting.graphs as qlg


class DORYExporter(qle.onnxexport.ONNXExporter):

    def __init__(self):
        annotator = DORYAnnotator()
        super(DORYExporter, self).__init__(annotator=annotator)

    @staticmethod
    def dump_features(network: nn.Module,
                      x:       torch.Tensor,
                      path:    os.PathLike) -> None:
        """Given a network, export the features associated with a given input.

        To verify the correctness of an ONNX export, DORY requires text files
        containing the values of the features for each layer in the target
        network.
        """

        class Features(NamedTuple):
            module_name: str
            features:    torch.Tensor

        # To validate the correctness of ONNX exports, DORY requires to dump
        # text files in a specific format:
        #
        #     https://github.com/pulp-platform/dory_examples/tree/master/examples/Quantlab_examples
        #
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
            assert (out_.ndim == 4) and (out_.shape[0] == 1)  # TODO: we are tacitly assuming that we will capture only features of 2D-conv networks (i.e., 4D feature arrays)
            # DORY wants HWC tensors
            features.append(Features(module_name=module_name, features=out_.squeeze(0)))

        # the core dump functionality starts here

        # 1. set up hooks to intercept features
        for n, m in network.named_modules():
            if isinstance(m, qlg.Requant):  # TODO: we are tacitly assuming that these layers will always output 4D feature arrays
                hook = partial(hook_fn, module_name=n)
                m.register_forward_hook(hook)

        # 2. propagate the supplied input through the network; the hooks will capture the features
        x = x.clone()
        y = network(x.to(dtype=torch.float32))

        # 3. export input, features, and output to text files
        export_to_txt('input', 'input', x)
        for i, (module_name, f) in enumerate(features):
            export_to_txt(module_name, f"out_layer{i}", f)
        export_to_txt('output', 'output', y)
