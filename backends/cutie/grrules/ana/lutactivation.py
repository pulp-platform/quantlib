import torch
import torch.nn as nn


class LUTActivation(nn.Module):
    
    def __init__(self, tau, quant_levels):
        super(LUTActivation, self).__init__()
        self.setup_parameters(self, tau, quant_levels)
        self.unfolded_tau = False
    
    @staticmethod
    def setup_parameters(lutmod, tau, quant_levels):
        lutmod.register_parameter('tau', nn.Parameter(tau, requires_grad=False))
        lutmod.register_parameter('quant_levels', nn.Parameter(quant_levels, requires_grad=False))
        lutmod.register_parameter('q0', nn.Parameter(quant_levels[0], requires_grad=False))
        lutmod.register_parameter('jumps', nn.Parameter(quant_levels[1:] - quant_levels[:-1], requires_grad=False))
        
    def forward(self, x):
        
        if not self.unfolded_tau:
            self.tau.data     = self.tau[(...,) + (None,) * (x.dim() - 2)]
            self.unfolded_tau = True
            
        x = x.unsqueeze(1)
        cdf = (x - self.tau >= 0.0).float()
        
        y = self.q0 + torch.sum(self.jumps[(...,) + (None,) * (cdf.dim() - 2)] * cdf, 1)

        return y
