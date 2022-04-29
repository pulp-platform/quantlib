import torch.nn as nn
from typing import Dict, Type


ModuleMapping = Dict[Type[nn.Module], Type[nn.Module]]
