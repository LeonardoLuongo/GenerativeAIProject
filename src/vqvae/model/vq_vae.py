import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.LeakyReLU(), 
            nn.Conv2d(dim, dim, 3, padding=1), 
            nn.GroupNorm(32, dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(self._num_embeddings, self._embedding_dim)
        self.register_buffer('_embedding', embedding)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', embedding.clone())

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        if self.training:
            dw = torch.sum(encodings, dim=0)
            self._ema_cluster_size.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            dw_embed = torch.matmul(encodings.t(), flat_input)
            self._ema_w.data.mul_(self.decay).add_(dw_embed, alpha=1 - self.decay)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = ((self._ema_cluster_size.data + self.epsilon) / (n + self._num_embeddings * self.epsilon) * n)
            self._embedding.data.copy_(self._ema_w.data / self._ema_cluster_size.data.unsqueeze(1))

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = F.embedding(encoding_indices, self._embedding).view(input_shape)
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encoding_indices

    def restart_unused_codes(self, inputs, threshold=1.0):
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self._embedding_dim)
        dead_indices = (self._ema_cluster_size < threshold).nonzero(as_tuple=True)[0]
        if len(dead_indices) > 0:
            num_samples = min(len(dead_indices), flat_input.size(0))
            rand_idx = torch.randperm(flat_input.size(0), device=inputs.device)[:num_samples]
            new_centers = flat_input[rand_idx].detach()
            with torch.no_grad():
                self._embedding[dead_indices[:num_samples]] = new_centers
                self._ema_w[dead_indices[:num_samples]] = new_centers
                self._ema_cluster_size[dead_indices[:num_samples]] = threshold
            return len(dead_indices)
        return 0

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=64, commitment_cost=0.25, decay=0.99):
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # --- ENCODER (Target: 16x16 Output) ---
        self.encoder = nn.Sequential(
            # Layer 1: 64x64 -> 32x32 (Stride 2)
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(),
            ResidualBlock(64),

            # Layer 2: 32x32 -> 16x16 (Stride 2)
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(),
            ResidualBlock(128),
            
            # Layer 3: 16x16 -> 16x16 (MODIFICA: Stride 1)
            # Abbiamo cambiato stride=2 in stride=1 e kernel=4 in kernel=3
            nn.Conv2d(128, embedding_dim, 3, stride=1, padding=1),
            nn.GroupNorm(32, embedding_dim),
            nn.LeakyReLU(),
            ResidualBlock(embedding_dim),
            
            # Pre-Quantization Refinement
            nn.Conv2d(embedding_dim, embedding_dim, 3, stride=1, padding=1),
        )

        self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)

        # --- DECODER (Target: 16x16 Input -> 64x64 Output) ---
        self.decoder = nn.Sequential(
            # Post-Quantization
            nn.Conv2d(embedding_dim, 128, 3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(), 
            ResidualBlock(128),
            
            # Layer 1: 16x16 -> 16x16 (MODIFICA: Mirroring Encoder)
            # Usiamo Conv2d invece di ConvTranspose perché non dobbiamo ingrandire qui
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(),
            ResidualBlock(64),
            
            # Layer 2: 16x16 -> 32x32 (Upsample Stride 2)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(),
            ResidualBlock(32),
            
            # Output Layer: 32x32 -> 64x64 (Upsample Stride 2)
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh() 
        )

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, vq_loss, perplexity, _ = self.vq_layer(z)
        x_recon = self.decoder(z_quantized)
        return x_recon, vq_loss, perplexity

    def get_codebook_indices(self, x):
        z = self.encoder(x)
        _, _, _, indices = self.vq_layer(z)
        return indices.view(x.shape[0], -1)
    
    def decode_from_indices(self, indices):
        z_q = F.embedding(indices, self.vq_layer._embedding)
        b, seq_len, dim = z_q.shape
        # Logica automatica: se seq_len=256 -> h=w=16. Se 64 -> h=w=8.
        h = w = int(seq_len**0.5) 
        z_q = z_q.permute(0, 2, 1).view(b, dim, h, w)
        return self.decoder(z_q)
    
    def restart_codebook(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            restarted_count = self.vq_layer.restart_unused_codes(z)
        return restarted_count