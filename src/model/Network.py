import torch.nn as nn
from TimeEncoding import *
from UNetBlock import *

TIME_ENCODING_SIZE = 256
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3    # RGB
NUM_ATTRIBUTES = 3
BASE_CHANNELS = 128


class ConditionalNetwork(nn.Module):
    def __init__(self, L=1000):
        super().__init__()
        
        #Creazione del time encoding
        self.time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

        # Input: 3 canali (RGB). Output: 128 feature iniziali
        self.pre = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, BASE_CHANNELS, 3, padding='same'),
            nn.SiLU()
        )
        
        
        # Struttura U-Net: 128 -> 256 -> 512 -> 1024
        self.unet = self.build_unet(IMAGE_SIZE, [BASE_CHANNELS, 256, 512, 1024])
        
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
            nn.Linear(128, TIME_ENCODING_SIZE), # Output deve essere 256
            nn.SiLU(),
        )

    def forward(self, x, t, attributes):
        #Ottieniamo encoding del Tempo
        t_enc = self.time_encoding[t] # Shape: (Batch, 64)
        
        #Ottieniamo encoding degli Attributi
        attr_enc = self.attribute_encoder(attributes.float()) # Shape: (Batch, 64)
        
        # Qui uniamo le informazioni. La rete riceve un vettore che contiene sia "quando" (t) che "cosa" (attr)
        combined_enc = t_enc + attr_enc
        
        # Passaggio nella rete
        x = self.pre(x) #Da 3 a 128 canali
        y = self.unet(x, combined_enc) # Passiamo l'embedding combinato
        y = self.post(y) #Da 128 a 3 canali
        
        return y

    def build_unet(self, size, feat_list):
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:]) #Chiamata ricorsiva, size//2 poichè ad ogni passo l'immagine diventa la metà 64->32
            if size <= 16:
                use_attn = True
            else:
                use_attn = False      
        else:
            inner_block = None
            use_attn = True # Attiviamo attention nel bottleneck (il livello più profondo 8x8)
        return UNetBlock(size, feat_list[0], feat_list[1], inner_block, use_attention=use_attn)