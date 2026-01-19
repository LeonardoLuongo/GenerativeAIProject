import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
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
        return self.relu(x + self.conv(x))

class CelebACVAE(nn.Module):
    def __init__(self, latent_dim, num_attr):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attr = num_attr

        # --- ENCODER ---
        # Input: (3 + num_attr) x 64 x 64
        self.enc_init = nn.Sequential(
            nn.Conv2d(3 + num_attr, 64, 4, stride=2, padding=1), # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64)
        )
        
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128)
        )

        self.enc_down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256)
        )

        self.enc_down3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_log_sigma = nn.Linear(512 * 4 * 4, latent_dim)

        # --- DECODER  ---
        self.dec_input = nn.Linear(latent_dim + num_attr, 512 * 4 * 4)
        
        # Ogni blocco riceve (Canali precedenti + num_attr)
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
            nn.Sigmoid()
        )

    def _inject_cond(self, x, cond):
        """Helper per iniettare gli attributi nelle mappe di caratteristiche"""
        c = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return torch.cat([x, c], dim=1)

    def forward(self, x, cond):
        # Encoding
        econd = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x_in = torch.cat((x, econd), dim=1)
        
        out = self.enc_init(x_in)
        out = self.enc_down1(out)
        out = self.enc_down2(out)
        out = self.enc_down3(out)
        out = self.flatten(out)
        
        mu = self.fc_mu(out)
        log_sigma = self.fc_log_sigma(out)
        
        # Reparameterization
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decoding con Deep Conditioning
        z_cond = torch.cat((z, cond), dim=1)
        y = self.dec_input(z_cond).view(-1, 512, 4, 4)
        
        y = self.up1(self._inject_cond(y, cond))
        y = self.up2(self._inject_cond(y, cond))
        y = self.up3(self._inject_cond(y, cond))
        y = self.up4(self._inject_cond(y, cond))
        
        return y, mu, log_sigma
