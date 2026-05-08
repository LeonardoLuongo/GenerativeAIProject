import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Blocco Residuo: preserva i dettagli ad alta frequenza nel bottleneck latente.
    Permette al gradiente di fluire attraverso le connessioni skip (x + f(x)).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Somma l'input originale all'output delle convoluzioni (Skip Connection)
        return self.relu(x + self.conv(x))

class CelebACVAE(nn.Module):
    """Conditional Variational Autoencoder con Deep Conditioning e Residual Blocks"""
    def __init__(self, latent_dim, num_attr):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attr = num_attr

        # --- ENCODER ---
        # Input: (3 canali RGB + num_attr) x 64 x 64
        self.enc_init = nn.Sequential(
            nn.Conv2d(3 + num_attr, 64, 4, stride=2, padding=1), # Out: [B, 64, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64)
        )
        
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # Out: [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128)
        )

        self.enc_down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),         # Out: [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256)
        )

        self.enc_down3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),         # Out: [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_log_sigma = nn.Linear(512 * 4 * 4, latent_dim)

        # --- DECODER ---
        # Input: Vettore latente concatenato agli attributi (z + num_attr)
        self.dec_input = nn.Linear(latent_dim + num_attr, 512 * 4 * 4)
        
        # Deep Conditioning: iniettiamo gli attributi ad ogni upsampling.
        # Evitiamo ConvTranspose2d usando Upsample + Conv2d per eliminare 
        # i 'checkerboard artifacts' (artefatti a scacchiera).
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 8x8
            nn.Conv2d(512 + num_attr, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 16x16
            nn.Conv2d(256 + num_attr, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32x32
            nn.Conv2d(128 + num_attr, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64x64
            nn.Conv2d(64 + num_attr, 3, 3, padding=1),
            nn.Sigmoid() # Sigmoid finale per mappare l'output nel range [0, 1] richiesto dalla BCE
        )

    def _inject_cond(self, x, cond):
        """
        Invece di dare gli attributi solo all'inizio, li 'spalmiamo' nello spazio.
        1. Prende cond [Batch, Num_Attr]
        2. Lo espande a [Batch, Num_Attr, Altezza, Larghezza] per matchare la feature map x
        3. Concatena lungo i canali.
        """
        c = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return torch.cat([x, c], dim=1)

    def forward(self, x, cond):
        # --- ENCODER ---
        # Concateniamo gli attributi all'immagine di input (64x64x(3+N))
        econd = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x_in = torch.cat((x, econd), dim=1)
        
        out = self.enc_down3(self.enc_down2(self.enc_down1(self.enc_init(x_in))))
        out = self.flatten(out)
        
        mu, log_sigma = self.fc_mu(out), self.fc_log_sigma(out)
        
        # Reparameterization Trick
        # Ciò rende la derivata calcolabile attraverso un campionamento stocastico
        z = mu + torch.randn_like(log_sigma) * torch.exp(0.5 * log_sigma)
        
        # --- DECODER ---
        # Iniziamo la ricostruzione partendo dal vettore latente z + attributi
        y = self.dec_input(torch.cat((z, cond), dim=1)).view(-1, 512, 4, 4)
        y = self.up1(self._inject_cond(y, cond))
        y = self.up2(self._inject_cond(y, cond))
        y = self.up3(self._inject_cond(y, cond))
        y = self.up4(self._inject_cond(y, cond))

        return y, mu, log_sigma