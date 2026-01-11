import torch
from torch import nn

class CelebACVAE(nn.Module):
    def __init__(self, latent_dim, num_attr):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attr = num_attr
        
        # Encoder: Input 64x64x(3 + num_attr)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3 + num_attr, 32, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_sigma = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: Latent + Cond -> 64x64x3
        self.decoder_input = nn.Linear(latent_dim + num_attr, 256 * 4 * 4)
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 64x64
            nn.Sigmoid() 
        )

    def forward(self, x, cond):
        # Espansione attributi per l'encoder
        econd = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x_cond = torch.cat((x, econd), dim=1)

        out = self.encoder_conv(x_cond)
        mu = self.fc_mu(out)
        log_sigma = self.fc_log_sigma(out)

        # Reparameterization
        eps = torch.randn_like(log_sigma)
        z = mu + eps * torch.exp(log_sigma)

        # Decoding
        z_cond = torch.cat((z, cond), dim=1)
        y = self.decoder_input(z_cond)
        y = self.decoder_conv(y)
        
        return y, mu, log_sigma