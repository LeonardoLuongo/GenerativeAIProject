import torch
import torch.nn as nn
from model.joint_embeddings import JointEmbeddings
from model.mamba.mamba_generator import MambaGenerator
from model.prediction_head import PixelClassifier

class MambaImageModel(nn.Module):
    def __init__(self, dim, state_size, expansion, conv_kernel, num_layers, num_attr=3):
        super().__init__()
        self.dim = dim
        self.num_attr = num_attr
        
        # 1. Input Layer
        self.embeddings = JointEmbeddings(dim_D=dim, num_attribute=num_attr)
        
        # 2. Motore Mamba
        self.mamba = MambaGenerator(
            dim=dim, 
            state_size=state_size, 
            expansion=expansion, 
            conv_kernel=conv_kernel, 
            num_layers=num_layers
        )
        
        # 3. Output Layer
        self.classifier = PixelClassifier(dim_D=dim, num_classes=256)

    def forward(self, images, attributes):
        """
        LOGICA DI TRAINING (Teacher Forcing)
        images: (B, 4096, 1) -> Pixel reali normalizzati [0, 1]
        attributes: (B, 3)   -> Int [0, 1]
        """
        # --- SHIFT PER TEACHER FORCING ---
        # Passiamo i pixel dallo 0 al 4094 per predire dall'1 al 4095
        input_pixels = images[:, :-1, :] # (B, 4095, 1)
        
        # 1. Embedding (Prefissi + Pixel) -> (B, 4098, D)
        x = self.embeddings(input_pixels, attributes)
        
        # 2. Elaborazione Mamba -> (B, 4098, D)
        x = self.mamba(x)
        
        # 3. Classificazione (Slicing)
        # Prendiamo gli ultimi 4096 stati per predire i 4096 pixel target
        # (L'ultimo attributo predice il pixel 0, il penultimo pixel predice l'ultimo)
        logits = self.classifier(x[:, (self.num_attr-1):, :]) 
        
        return logits # (B, 4096, 256)

    @torch.no_grad()
    def generate(self, attributes):
        """
        LOGICA DI INFERENZA (Auto-regressiva)
        Genera l'immagine un pixel alla volta partendo dagli attributi.
        """
        device = attributes.device
        B = attributes.shape[0]
        
        # Reset delle cache h del professore
        self.mamba.inference_start(B)
        
        # 1. STEP ATTRIBUTI: "Carichiamo" la memoria di Mamba con le condizioni
        attr_emb = self.embeddings.attr_emb(attributes.long()) # (B, 3, D)
        for i in range(self.num_attr):
            self.mamba.inference_step(attr_emb[:, i:i+1, :])
            
        # 2. STEP GENERAZIONE PIXEL
        generated_indices = []
        # Partiamo da un pixel "nero" (valore 0)
        last_pixel_val = torch.zeros(B, 1, 1, device=device) 
        
        # Recuperiamo i positional encoding pre-calcolati (dal modulo embeddings)
        pos_enc_all = self.embeddings.create_2d_pos_encoding(dim=64, device=device)
        
        for i in range(4096):
            # Proiezione del pixel corrente + il suo encoding di posizione i-esimo
            pix_proj = self.embeddings.pixel_projection(last_pixel_val) # (B, 1, D)
            pix_token = pix_proj + pos_enc_all[i:i+1, :]
            
            # Passaggio nel motore Mamba (aggiorna la cache interna)
            mamba_out = self.mamba.inference_step(pix_token) # (B, 1, D)
            
            # Classificazione e scelta del pixel (Argmax)
            logits = self.classifier.linear_head(mamba_out) # (B, 1, 256)
            pixel_id = torch.argmax(logits, dim=-1) # (B, 1)
            
            generated_indices.append(pixel_id)
            
            # Prepara l'input per il prossimo step: normalizza l'indice in [0, 1]
            last_pixel_val = pixel_id.float().unsqueeze(-1) / 255.0
            
        return torch.cat(generated_indices, dim=1) # (B, 4096)