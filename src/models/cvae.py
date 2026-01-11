import torch
from torch import nn

class CelebACVAE(nn.Module):
    def __init__(self, latent_dim, num_attr):
        super().__init__()
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3 + num_attr, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), # Stabilizza il gradiente
            nn.LeakyReLU(0.2),  # Meglio di ReLU per le facce
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_sigma = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + num_attr, 256 * 4 * 4)
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x, cond):
        econd = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x_cond = torch.cat((x, econd), dim=1)
        out = self.encoder_conv(x_cond)
        mu = self.fc_mu(out)
        log_sigma = self.fc_log_sigma(out)
        eps = torch.randn_like(log_sigma)
        z = mu + eps * torch.exp(log_sigma)
        z_cond = torch.cat((z, cond), dim=1)
        y = self.decoder_input(z_cond)
        y = self.decoder_conv(y)
        return y, mu, log_sigma