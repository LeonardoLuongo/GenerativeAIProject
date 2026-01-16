import torch
import torch.nn as nn
import torch.nn.functional as F
# Importiamo la classe del prof
from Attention import SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_ENCODING_SIZE = 128

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Inizializzazione a zero per stabilità
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h

class UNetBlock(nn.Module):
    # Aggiunto parametro use_attention
    def __init__(self, size, outer_features, inner_features, inner_block=None, use_attention=False):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.inner = inner_block
        self.use_attention = use_attention

        self.encoder = self.build_encoder(outer_features, inner_features)
        
        # --- BOTTLENECK & ATTENTION ---
        if self.use_attention:
            # 'dim' corrisponde al numero di canali (inner_features)
            self.attn_enc = SelfAttention(dim=inner_features, heads=4)
            self.attn_dec = SelfAttention(dim=outer_features, heads=4)
            # Ci serve una norma per stabilizzare l'attenzione
            self.norm_enc = nn.GroupNorm(8, inner_features)
            self.norm_dec = nn.GroupNorm(8, outer_features)
        # Bottleneck (solo se siamo al centro)
        if inner_block is None:
            self.bottleneck = ResidualBlock(inner_features)
        else:
            self.bottleneck = None
        
        self.decoder = self.build_decoder(inner_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)

    def apply_attention(self, x, attention_layer, norm_layer):
        """Funzione helper per applicare l'attenzione su tensori immagine"""
        B, C, H, W = x.shape # x shape: (Batch, Canali, H, W)
        y_in = norm_layer(x) # Normalizziamo prima dell'attenzione
        # Dove L = H*W (Lunghezza sequenza) e D = C (Embedding)
        # .view(B, C, -1) crea (B, C, H*W)
        # .permute(0, 2, 1) scambia la dimensione C con la dimensione H*W perche cosi serve sia  
        # l'ornine nella classe attention
        y_seq = y_in.view(B, C, -1).permute(0, 2, 1) # (B, L, C)
        y_attn = attention_layer(y_seq)  # Chiamiamo l'attention per capire il contesto globale per ogni pixel
        y_attn = y_attn.permute(0, 2, 1).view(B, C, H, W) #Ritrasformiamo in immagine: (B, L, D) -> (B, C, H, W)
        return x + y_attn #Somma Residua (Skip Connection dell'attenzione)

    def forward(self, x, context_embedding):
        # 1. Discesa
        y = self.encoder(x)
        
        # APPLICAZIONE ATTENZIONE ENCODER (se attivata)
        # Viene applicata dopo il downsampling dell'encoder
        if self.use_attention:
            y = self.apply_attention(y, self.attn_enc, self.norm_enc)
        
        # 2. Centro
        if self.inner:
            y = self.inner(y, context_embedding)
        else:
            # Siamo al Bottleneck
            y = self.bottleneck(y) #qui l'attenzione è già stata applicata dallo step precedente (attn_enc)
            # poiché nel bottleneck size=8, l'encoder ha portato a size=8 e attn_enc ha agito lì.
            
        # 3. Risalita 
        context_reshaped = context_embedding.view(-1, TIME_ENCODING_SIZE, 1, 1)# trasformo il vettore di contesto in un vettore 64x1x1
        half_size = self.size // 2
        y1 = torch.cat((y, context_reshaped.expand(-1, TIME_ENCODING_SIZE, half_size, half_size)), dim=1)#con expand creo una matrice 8x8x64 dove copio il vettore per ogni punto dell 8x8
        x1 = self.decoder(y1)
        
        #skip connection
        x2 = torch.cat((x1, x), dim=1)
        out = self.combiner(x2)

        if self.use_attention:
            out = self.apply_attention(out, self.attn_dec, self.norm_dec)

        return out

    def build_combiner(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, to_features, 3, padding=1),
            ResidualBlock(to_features)
        )

    def build_encoder(self, from_features, to_features):
        return nn.Sequential(
            ResidualBlock(from_features),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1)
        )

    def build_decoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding=1), 
            nn.GroupNorm(8, from_features),
            nn.SiLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1)
        )