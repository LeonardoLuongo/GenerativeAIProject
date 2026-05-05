import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeEncoding(nn.Module):
    def __init__(self, L, dim, device=device):
        super().__init__()
        self.L = L
        self.dim = dim
        dim2 = dim // 2
        encoding = torch.zeros(L, dim)
        ang = torch.linspace(0.0, torch.pi / 2, L) #valore da 0 a L ma convertito in un intervallo continuo 0-pi/2
        logmul = torch.linspace(0.0, math.log(100), dim2) # 100 è arbitrario, va scelto il base al numero L di step
        mul = torch.exp(logmul)
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        self.register_buffer('encoding', encoding.to(device)) # register_buffer salva lo stato ma non lo allena

    def forward(self, t):
        # t deve essere un tensore di indici (es. [50, 999, 0...])
        return self.encoding[t]
    
    def __getitem__(self, t):
        return self.encoding[t]