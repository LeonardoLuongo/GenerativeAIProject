import torch
from torch import nn


class JointEmbeddings(nn.Module):
    def __init__(self, dim_D:int, num_attribute:int=3):
        super().__init__()

        # Proiezione per il singolo valore del pixel
        self.pixel_projection = nn.Linear(in_features=1, out_features=dim_D)

        # Embedding per gli attributi (0 o 1)
        # Creiamo un "vocabolario" di 2 stati per ogni attributo
        self.attr_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_D)
        
        # Positional Encodings Appresi (2D)
        # 64 righe e 64 colonne, ognuna con metà della dimensione D
        self.row_emb = nn.Embedding(num_embeddings=64, embedding_dim=dim_D//2)
        self.col_emb = nn.Embedding(num_embeddings=64, embedding_dim=dim_D//2)

    def create_2d_pos_encoding(self, dim, device):
        # Genera una griglia (64x64) di indici di riga e colonna
        grid_rows = torch.arange(dim, device=device).repeat_interleave(dim)
        grid_cols = torch.arange(dim, device=device).repeat(dim)

        # Estrae i vettori corrispondenti
        row_emb = self.row_emb(grid_rows) # (4096, D/2)
        col_emb = self.col_emb(grid_cols) # (4096, D/2)

        # Concatena per formare l'encoding spaziale completo
        return torch.cat([row_emb, col_emb], dim=-1) # (4096, D)
    
    def forward(self, pixel_seq, attr_vector):
        """
        Input: 
          pixel_seq: (B, 4095, 1)
          attr_vector: (B, 3) -> valori interi 0 o 1
        """
        # 0. Metti tutto sullo stesso device
        device = pixel_seq.device

        # 1. Proietta i pixel nello spazio latente D
        x_vis = self.pixel_projection(pixel_seq) # Out: (B, 4096, D)

        # 2. Aggiungi l'informazione spaziale
        pos_enc = self.create_2d_pos_encoding(dim=64, device=device) # Out: (4096, D)
        pos_enc = pos_enc[None, :, :].repeat(pixel_seq.shape[0], 1, 1) # Out: (B, 4096, D)
        x_vis = x_vis + pos_enc[:, :-1, :]                 # Broadcasting: (B, 4095, D)

        # 3. Proietta gli attributi nello spazio latente D
        x_attr = self.attr_emb(attr_vector)     # Out: (B, 3, D)

        # 4. Concatena i due embeddings
        full_sequence = torch.concat((x_attr, x_vis), dim=1)

        return full_sequence # Out: (B, 4099, D)