import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURAZIONE ---
EXP_NAME = "cvae_prof_style_sum_loss_encdecBIG128s"
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
import config
from src.models.cvae import CelebACVAE

# --- PARAMETRI ---
BATCH_SIZE = 256     
LATENT_DIM = 128    
LR = 1e-4
EPOCHS = 100          
IMG_SIZE = 64
BETA = 1.0           
ATTR_INDICES = [20, 31, 39] 
NUM_ATTR = len(ATTR_INDICES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FUNZIONI LOSS ---
reconstruction_loss_function = nn.BCELoss(reduction='sum')

def kl_loss_function(mu, log_sigma):
    kl = 0.5 * (mu**2 + torch.exp(2 * log_sigma) - 1 - 2 * log_sigma)
    return torch.sum(kl)

def train():
    model_dir = config.MODEL_PATH / EXP_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.TRAIN_RES_PATH / EXP_NAME)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CelebA(root=str(config.DATASET_BASIC_PATH), split='train', target_type='attr', transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"--- START TRAINING ---")
    
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, attr in progress_bar:
            cond = attr[:, ATTR_INDICES].float().to(DEVICE)
            x = x.to(DEVICE)

            optimizer.zero_grad()
            output, mu, log_sigma = model(x, cond)
            
            # LOSS CALCOLATA COME SOMMA
            recon_loss = reconstruction_loss_function(output, x)
            kl_loss = kl_loss_function(mu, log_sigma)
            
            loss = recon_loss + BETA * kl_loss
            
            loss.backward()
            optimizer.step()
            
            # Logging
            total_epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.0f}"})

            if global_step % 10 == 0:
                writer.add_scalar('Loss/Total_Sum', loss.item(), global_step)
                writer.add_scalar('Loss/Recon_Sum', recon_loss.item(), global_step)
                writer.add_scalar('Loss/KL_Sum', kl_loss.item(), global_step)

            if global_step % 500 == 0:
                with torch.no_grad():
                    model.eval()
                    comparison = torch.cat([x[:8], output[:8]])
                    grid = vutils.make_grid(comparison, nrow=8, normalize=False)
                    writer.add_image('Reconstruction', grid, global_step)
                    model.train()
            
            global_step += 1

        print(f"===> Epoch {epoch} completed. Avg Loss: {total_epoch_loss / len(train_loader):.0f}")
        torch.save(model.state_dict(), str(model_dir / "model.pth"))

    writer.close()

if __name__ == "__main__":
    train()