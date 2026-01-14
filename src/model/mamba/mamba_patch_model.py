import torch
import torch.nn as nn
from model.patch_components import PatchEmbedding, PatchReconstructor, DeblockingHead
from model.mamba.mamba_generator import MambaGenerator

class PatchMambaModel(nn.Module):
    def __init__(self, 
                 dim=512, 
                 state_size=16, 
                 expansion=1.1, 
                 conv_kernel=4, 
                 num_layers=5, 
                 img_size=64, 
                 patch_size=4, 
                 channels=3, 
                 num_attr=3,
                 latent_dim=128): # Nuovo parametro per il rumore
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_attr = num_attr
        self.latent_dim = latent_dim # Dimensione del vettore rumore
        
        self.h_patches = img_size // patch_size
        self.num_patches = self.h_patches ** 2
        self.patch_dim = channels * patch_size * patch_size
        
        # 1. SOS Token Apprendibile
        self.sos_token = nn.Parameter(torch.randn(1, 1, self.patch_dim))
        
        # 2. Componenti (passiamo latent_dim)
        self.embedding = PatchEmbedding(dim, self.patch_dim, self.num_patches, num_attr, latent_dim)
        self.mamba = MambaGenerator(dim, state_size, expansion, conv_kernel, num_layers)
        self.reconstructor = PatchReconstructor(dim, self.patch_dim)
        self.deblocker = DeblockingHead(channels)

    def patchify(self, images):
        B, C, H, W = images.shape
        P = self.patch_size
        patches = images.view(B, C, H//P, P, W//P, P)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, -1, self.patch_dim)
        return patches

    def unpatchify(self, patches):
        B = patches.shape[0]
        P = self.patch_size
        C = self.channels
        H = W = self.img_size
        patches = patches.view(B, H//P, W//P, C, P, P)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = patches.view(B, C, H, W)
        return images

    def forward(self, images, attributes):
        """
        TRAINING
        """
        B = images.shape[0]
        device = images.device
        
        # 1. Patch Reali
        real_patches = self.patchify(images) 
        
        # 2. Prepariamo Input Sequenza con SOS
        sos_batch = self.sos_token.expand(B, -1, -1) 
        input_patches = torch.cat([sos_batch, real_patches[:, :-1, :]], dim=1) 
        
        # 3. GENERAZIONE RUMORE (Diversità durante il training)
        # Ogni volta che l'immagine passa, riceve un rumore leggermente diverso
        # Questo costringe il modello a non ignorare il rumore
        z = torch.randn(B, self.latent_dim, device=device)
        
        # 4. Mamba (con iniezione rumore)
        x = self.embedding(input_patches, attributes, z)
        x = self.mamba(x)
        
        # 5. Ricostruzione
        predicted_patches = self.reconstructor(x) 
        
        # 6. Unpatchify e Deblocking
        coarse_image = self.unpatchify(predicted_patches)
        refined_image = self.deblocker(coarse_image)
        
        return predicted_patches, refined_image


    @torch.no_grad()
    def generate(self, attributes):
        """
        INFERENCE
        """
        B = attributes.shape[0]
        device = attributes.device
        
        self.mamba.inference_start(B)
        
        # 1. GENERAZIONE RUMORE (Diversità in inferenza)
        z = torch.randn(B, self.latent_dim, device=device)
        
        # 2. Calcolo vettore Stile (Attributi + Rumore)
        attr_emb = self.embedding.attr_emb(attributes.long()).view(B, -1)
        style_input = torch.cat([attr_emb, z], dim=1)
        style_vector = self.embedding.style_proj(style_input).unsqueeze(1) # (B, 1, D)
        
        generated_patches = []
        
        # STEP 0: SOS Token
        last_patch = self.sos_token.expand(B, -1, -1) 
        pos_enc = self.embedding.pos_emb.weight 
        
        # Loop su 256 patches (per 64x64 img con 4x4 patch)
        for i in range(self.num_patches):
            # Proiezione Patch
            patch_proj = self.embedding.patch_proj(last_patch)
            
            # Aggiunta Posizione
            curr_pos = pos_enc[i:i+1, :].unsqueeze(0)
            x = patch_proj + curr_pos
            
            # AGGIUNTA STILE (Attributi + Rumore)
            x = x + style_vector
            
            # Mamba Step
            out_emb = self.mamba.inference_step(x)
            
            # Ricostruzione
            rec_patch = self.reconstructor(out_emb)
            generated_patches.append(rec_patch)
            
            # Next step input (Autoregressione)
            last_patch = rec_patch
            
        all_patches = torch.cat(generated_patches, dim=1)
        raw_image = self.unpatchify(all_patches)
        final_image = self.deblocker(raw_image)
        
        return final_image