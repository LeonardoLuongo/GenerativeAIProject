import torch
from torch import nn

class PixelClassifier(nn.Module):
    def __init__(self, dim_D:int, num_classes:int = 256):
        super().__init__()
        self.linear_head = nn.Linear(in_features=dim_D, out_features=num_classes)

    def forward(self, mamba_output, device=None):
        if device is None:
            device = mamba_output.device.type

        # Trasformiamo la rappresentazione astratta in probabilità per 256 livelli di grigio
        logits = self.linear_head(mamba_output)     # Out: (B, 4096, 256)

        return logits