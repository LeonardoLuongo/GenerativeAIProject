import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, dim_D: int, patch_dim: int, num_patches: int, num_attr: int = 3, latent_dim: int = 128):
        super().__init__()
        self.patch_dim = patch_dim
        
        # Proiezione del patch visivo
        self.patch_proj = nn.Linear(patch_dim, dim_D)
        
        # Embedding degli attributi + RUMORE (Noise Injection)
        # 1. Embeddiamo gli attributi binari (0/1) in vettori di dimensione D
        self.attr_emb = nn.Embedding(2, dim_D)
        
        # 2. Proiezione dello stile: Prende [Attributi + Rumore] e crea lo stile globale
        # Input: (3 attributi * D) + (Latent Noise Dim)
        input_style_dim = (num_attr * dim_D) + latent_dim
        self.style_proj = nn.Linear(input_style_dim, dim_D)
        
        # Positional Embedding
        self.pos_emb = nn.Embedding(num_patches + 1, dim_D) 

    def forward(self, patch_seq, attr_vector, z_noise):
        """
        patch_seq: (B, L, Patch_Dim)
        attr_vector: (B, 3)
        z_noise: (B, latent_dim) -> IL FATTORE CAOS
        """
        B, L, _ = patch_seq.shape
        device = patch_seq.device

        # 1. Embed dei Patch
        x = self.patch_proj(patch_seq) # (B, L, D)
        
        # 2. Embed Stile (Attributi + Rumore)
        # Embed attributes: (B, 3) -> (B, 3, D) -> Flat (B, 3*D)
        attrs_flat = self.attr_emb(attr_vector.long()).view(B, -1)
        
        # Concateniamo il rumore agli attributi!
        # [ATTR_1_EMB, ATTR_2_EMB, ATTR_3_EMB, NOISE_VECTOR]
        style_input = torch.cat([attrs_flat, z_noise], dim=1) # (B, 3*D + latent)
        
        # Proiettiamo tutto in un unico vettore "Stile" di dimensione D
        style_vector = self.style_proj(style_input) # (B, D)
        
        # 3. Positional Encoding
        positions = torch.arange(L, device=device).expand(B, L)
        x = x + self.pos_emb(positions)

        # 4. INIEZIONE GLOBALE
        # Sommiamo lo stile (che ora contiene il rumore unico per questa immagine)
        x = x + style_vector.unsqueeze(1)
        
        return x

class PatchReconstructor(nn.Module):
    def __init__(self, dim_D: int, patch_dim: int):
        super().__init__()
        # Aggiungiamo un po' di profondità qui visto che le patch sono piccole (4x4)
        # per aiutare a decodificare meglio i dettagli
        self.net = nn.Sequential(
            nn.Linear(dim_D, dim_D),
            nn.GELU(),
            nn.Linear(dim_D, patch_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class DeblockingHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, channels, kernel_size=3, padding='same'),
            nn.Tanh() 
        )

    def forward(self, img):
        delta = self.residual_net(img)
        refined = torch.clamp(img + delta, 0.0, 1.0)
        return refined