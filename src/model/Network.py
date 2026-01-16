import torch.nn as nn
from TimeEncoding import *
from UNetBlock import *
import torch

TIME_ENCODING_SIZE = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3    # RGB
NUM_ATTRIBUTES = 3
BASE_CHANNELS = 128


class ConditionalNetwork(nn.Module):
    def __init__(self, L=1000, cfg_dropout=0.1):
        super().__init__()
        
        self.cfg_dropout = cfg_dropout
        #Creazione del time encoding
        self.time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

        # Input: 3 canali (RGB). Output: 128 feature iniziali
        self.pre = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, BASE_CHANNELS, 3, padding='same'),
            nn.GroupNorm(8, BASE_CHANNELS),
            nn.SiLU()
        )
        
        
        # Struttura U-Net: 128 -> 256 -> 512 -> 1024
        self.unet = self.build_unet(IMAGE_SIZE, [BASE_CHANNELS, 256, 512, 768])
        
        # Riporta le feature a 3 canali (RGB)
        self.post = nn.Sequential(
            nn.GroupNorm(8, BASE_CHANNELS),
            nn.SiLU(),
            nn.Conv2d(BASE_CHANNELS, IMAGE_CHANNELS, 3, padding='same')
        )
        
        # Trasforma i 3 attributi (es. [1, 0, 1]) in un vettore grande come quello del tempo (256)
        self.attribute_encoder = nn.Sequential(
            nn.Linear(NUM_ATTRIBUTES, 128),     
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, TIME_ENCODING_SIZE) # Output deve essere 128
        )

         # MLP per combinare tempo + attributi (invece di somma semplice)
        self.context_combiner = nn.Sequential(
            nn.Linear(TIME_ENCODING_SIZE * 2, TIME_ENCODING_SIZE * 2),
            nn.SiLU(),
            nn.Linear(TIME_ENCODING_SIZE * 2, TIME_ENCODING_SIZE)
        )

        self._init_weights()

    def _init_weights(self):
        """Inizializzazione Xavier per stabilità"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t, attributes, drop_condition=False):
        #Ottieniamo encoding del Tempo
        t_enc = self.time_encoding[t]
        
        #Ottieniamo encoding degli Attributi
        if drop_condition:
            # Usa vettore zero per gli attributi
            attr_enc = torch.zeros_like(t_enc)
        else:
            # Training normale con possibile dropout casuale
            if self.training and torch.rand(1).item() < self.cfg_dropout:
                attr_enc = torch.zeros_like(t_enc)
            else:
                attr_enc = self.attribute_encoder(attributes.float())
        
        
        # Combiniamo le informazioni. La rete riceve un vettore che contiene sia "quando" (t) che "cosa" (attr)
        combined = torch.cat([t_enc, attr_enc], dim=1)
        context = self.context_combiner(combined)
        
        # Passaggio nella rete
        x = self.pre(x) #Da 3 a 128 canali
        x = self.unet(x, context) # Passiamo l'embedding combinato
        x = self.post(x) #Da 128 a 3 canali
        
        return x

    def build_unet(self, size, feat_list):
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:]) #Chiamata ricorsiva, size//2 poichè ad ogni passo l'immagine diventa la metà 64->32
            use_attn = (size <= 32)     
        else:
            inner_block = None
            use_attn = True # Attiviamo attention nel bottleneck (il livello più profondo 8x8)
        return UNetBlock(size, feat_list[0], feat_list[1], inner_block, use_attention=use_attn)