import torch
import torch.fx as fx

from quantlib.editing.editing.editors.nnmodules import NodesMap
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from quantlib.editing.editing.editors.nnmodules import NNModuleApplier


class WeightRounderApplier(NNModuleApplier):

    def __init__(self, pattern: NNSequentialPattern):
        super(WeightRounderApplier, self).__init__(pattern)

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        """Add rounding to the weights fake value.

        This operates on weights, calculating the rounding factor as one half of
        their `scale` and then applying it to the weights directly.
        After that, it reinits quantization hyperparameters.

        """

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_linear        = name_to_match_module['linear']

        # modify matched `nn.Module`s in-place
        rounding = (module_linear.scale.data.detach().clone()) * 0.5
        module_linear.weight.data  += torch.as_tensor(rounding, device=module_linear.weight.device)
        module_linear.init_qhparams()

        return g
