import torch
from torch import nn
from model.mamba.residual_mamba_layer import ResidualMambaLayer

class MambaGenerator(nn.Module):
    def __init__(self, dim, state_size, expansion, conv_kernel, num_layers):
        """
        Il generatore è lo stack di blocchi Mamba.
        Riceve sequenze di dimensione (B, L, D) e restituisce (B, L, D).
        """
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualMambaLayer(
                dim=dim, 
                state_size=state_size, 
                conv_kernel=conv_kernel, 
                expansion=expansion
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Passaggio standard (Parallel Scan) usato in Training.
        X: (B, L, D)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def inference_start(self, batch_size=1):
        """
        Prepara tutti i layer interni per la generazione step-by-step
        pulendo le cache degli stati precedenti (h).
        """
        for layer in self.layers:
            layer.inference_start(batch_size)

    def inference_step(self, x):
        """
        Passaggio sequenziale usato in Inferenza (generazione pixel per pixel).
        X: (B, 1, D)
        """
        for layer in self.layers:
            x = layer.inference_step(x)
        return x