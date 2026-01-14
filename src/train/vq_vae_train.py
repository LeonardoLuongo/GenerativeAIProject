import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.vq_vae.vq_vae import VQVAE

# --- CONFIGURAZIONE PER ALTA NITIDEZZA (16x16 Grid) ---
BATCH_SIZE = 256
EPOCHS = 50
EMBEDDING_DIMENSION = 1024 # Dimensione Vocabolario
LATENT_DIM = 256           # Canali del vettore latente (depth)
SOBEL_WEIGHT = 1         
COMMITMENT_COST = 0.5     
EMA_DECAY = 0.99          

class SobelLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.filter_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.filter_x = self.filter_x.repeat(3, 1, 1, 1)
        self.filter_y = self.filter_y.repeat(3, 1, 1, 1)

    def forward(self, pred, target):
        pred_dx = torch.nn.functional.conv2d(pred, self.filter_x, padding=1, groups=3)
        pred_dy = torch.nn.functional.conv2d(pred, self.filter_y, padding=1, groups=3)
        pred_grad = torch.sqrt(pred_dx**2 + pred_dy**2 + 1e-6)

        target_dx = torch.nn.functional.conv2d(target, self.filter_x, padding=1, groups=3)
        target_dy = torch.nn.functional.conv2d(target, self.filter_y, padding=1, groups=3)
        target_grad = torch.sqrt(target_dx**2 + target_dy**2 + 1e-6)

        return torch.nn.functional.mse_loss(pred_grad, target_grad)

def save_checkpoint(model, optimizer, epoch, filename="vqvae_checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # Normalizzazione [-1, 1] Tanh Ready
    transform = v2.Compose([
        v2.CenterCrop(178),
        v2.Resize((64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        training_set = CelebA(root='./dataset', split='train', target_type='attr', transform=transform, download=True)
    except RuntimeError as e:
        print(f"Errore dataset: {e}")
        return

    model = VQVAE(
        num_embeddings=EMBEDDING_DIMENSION,
        embedding_dim=LATENT_DIM,
        commitment_cost=COMMITMENT_COST,
        decay=EMA_DECAY
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mse_loss_fn = nn.MSELoss()
    sobel_loss_fn = SobelLoss(device)
    l1_loss_fn = nn.L1Loss()

    writer = SummaryWriter("runs/vqvae_celeba_16x16_2") # Cambio nome cartella log
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    global_step = 0
    print("Creazione batch fisso...")
    fixed_images, _ = next(iter(training_loader))
    fixed_images = fixed_images[:8].to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss_acc = 0.0
        perplexity_acc = 0.0

        loop = tqdm.tqdm(training_loader, leave=True)
        
        for batch_idx, (x, _) in enumerate(loop):
            x = x.to(device)
            
            # Restart Codebook (ogni 50 step)
            if batch_idx % 50 == 0:
                restarted = model.restart_codebook(x)
                if restarted > 0:
                    writer.add_scalar("Metrics/restarted_codes", restarted, global_step)

            optimizer.zero_grad()
            
            x_recon, vq_loss, perplexity = model(x)

            # 1. MSE Loss (Base)
            recon_loss = mse_loss_fn(x_recon, x)

            # 2. Sobel Loss (Nitidezza Bordi)
            s_loss = sobel_loss_fn(x_recon, x)

            # 3. L1 Loss (Colori e Robustezza)
            l1_loss = l1_loss_fn(x_recon, x)

            # Loss Mix
            loss = recon_loss + vq_loss + 2.0 * l1_loss + (SOBEL_WEIGHT * s_loss)

            loss.backward()
            optimizer.step()

            total_loss_acc += loss.item()
            perplexity_acc += perplexity.item()

            writer.add_scalar("Loss/total_step", loss.item(), global_step)
            writer.add_scalar("Metrics/perplexity_step", perplexity.item(), global_step)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            restart_msg = f" Rst:{restarted}" if batch_idx % 50 == 0 and restarted > 0 else ""
            loop.set_postfix(loss=loss.item(), perp=f"{perplexity.item():.1f}", sobel=f"{s_loss.item():.4f}" + restart_msg)

            global_step += 1

        scheduler.step()
        avg_loss = total_loss_acc / len(training_loader)
        avg_perplexity = perplexity_acc / len(training_loader)

        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        writer.add_scalar("Metrics/epoch_perplexity", avg_perplexity, epoch)

        print(f'End Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perp: {avg_perplexity:.1f}/{EMBEDDING_DIMENSION}')

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                recon_fixed, _, _ = model(fixed_images)
                indices = model.get_codebook_indices(fixed_images)
                recon_from_indices = model.decode_from_indices(indices)

            def denorm(img_tensor):
                return img_tensor * 0.5 + 0.5

            original_grid = torchvision.utils.make_grid(denorm(fixed_images.cpu()), nrow=4)
            recon_grid = torchvision.utils.make_grid(denorm(recon_fixed.cpu()), nrow=4)
            indices_grid = torchvision.utils.make_grid(denorm(recon_from_indices.cpu()), nrow=4)

            writer.add_image("Recon/Original", original_grid, epoch)
            writer.add_image("Recon/Standard", recon_grid, epoch)
            writer.add_image("Recon/FromIndices_MambaReady", indices_grid, epoch)
            
            if (epoch + 1) % 5 == 0: 
                save_checkpoint(model, optimizer, epoch, filename=f"checkpoints/vqvae_16x16_2_epoch_{epoch+1}.pth")

    writer.close()
    print("Training completato.")

if __name__ == "__main__":
    train()