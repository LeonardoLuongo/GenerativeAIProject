import torch
from torch import nn
from mamba.model.mamba_layer import MambaLayer

class ResidualMambaLayer(nn.Module):
    def __init__(self, dim, state_size, conv_kernel=4,
                 expansion=1):
        super().__init__()
        self.mamba=MambaLayer(dim, state_size, conv_kernel, expansion)
        self.norm=nn.LayerNorm(dim)

    def forward(self, x):
        return x+self.mamba(self.norm(x))

    def inference_start(self, batch_size=1):
        self.mamba.inference_start(batch_size)

    def inference_step(self, x):
        return x+self.mamba.inference_step(self.norm(x))
