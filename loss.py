import torch
import torch.nn as nn

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def get_bits(self,likelihood):
        """改为可处理多个likelihood"""
        bits=0

        if isinstance(likelihood,dict):
            for prob in likelihood:
                bits += -torch.sum(torch.log2(likelihood[prob]))
        else:
            bits = -torch.sum(torch.log2(likelihood))
        return bits

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = self.get_bits(output["likelihood"])/num_pixels
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        return out



