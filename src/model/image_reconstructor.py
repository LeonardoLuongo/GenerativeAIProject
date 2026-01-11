import torch
from torch import nn


class ImageReconstructor(nn.Module):
    def __init__(self):
        super().__init__() # Obbligatorio per nn.Module

    def forward(self, X):
        '''
        Prende l'output del classificatore e genera l'immagine

        :param X: Tensore di input con forma (B, 4096, 256)
        '''
        # 1. Trova l'indice del valore massimo (il "colore" del pixel)
        # Output: (B, 4096)
        pixel_idx = torch.argmax(X, dim=-1)

        # 2. Reshape per ottenere una griglia 64x64
        # Usiamo -1 per inferire automaticamente la dimensione del batch (B)
        img_grid = pixel_idx.view(-1, 64, 64)

        # Restituiamo il tensore. Se serve NumPy, lo convertiamo fuori.
        return img_grid
