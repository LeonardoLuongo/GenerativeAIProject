import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import warnings


EXP_NAME = "cvae_v2_mse_latent256_beta05"

# --- PARAMETRI ---
BATCH_SIZE = 128    
LATENT_DIM = 512     
LR = 1e-3
EPOCHS = 30          
IMG_SIZE = 64
BETA = 0.5           # Beta-VAE
ATTR_INDICES = [20, 31, 39] # Male, Smiling, Young
NUM_ATTR = len(ATTR_INDICES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- AGGIUNTA ROOT AL PATH ---
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

import config
from src.models.cvae import CelebACVAE

def kl_loss_function(mu, log_sigma):
    """Calcolo della KL Divergence"""
    kl = 0.5 * (mu**2 + torch.exp(2 * log_sigma) - 1 - 2 * log_sigma)
    return torch.sum(kl)

def train():
    model_dir = config.MODEL_PATH / EXP_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = config.TRAIN_RES_PATH / EXP_NAME
    writer = SummaryWriter(log_dir)


    # 2. Dataset con DATA AUGMENTATION (Flip orizzontale)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # Raddoppia virtualmente il dataset
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CelebA(
        root=str(config.DATASET_BASIC_PATH),
        split='train',
        target_type='attr',
        transform=transform,
        download=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )

    # 3. Inizializzazione Modello e Loss (MSE)
    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Passiamo a MSE per immagini RGB più nitide
    reconstruction_loss_fn = nn.MSELoss(reduction='sum')

    print(f"--- START TRAINING ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: Batch={BATCH_SIZE}, Latent={LATENT_DIM}, Beta={BETA}, Loss=MSE")

    # 4. Loop di Training
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
        
        for i, (x, attr) in progress_bar:
            cond = attr[:, ATTR_INDICES].float().to(DEVICE)
            x = x.to(DEVICE)

            optimizer.zero_grad()
            recon_x, mu, log_sigma = model(x, cond)
            
            # Calcolo Loss
            loss_recon = reconstruction_loss_fn(recon_x, x)
            loss_kl = kl_loss_function(mu, log_sigma)
            loss = loss_recon + BETA * loss_kl
            
            loss.backward()
            optimizer.step()
            
            # Logging progress bar
            current_loss = loss.item() / BATCH_SIZE
            progress_bar.set_postfix({"loss": f"{current_loss:.2f}"})

            # TensorBoard Scalari
            if global_step % 10 == 0:
                writer.add_scalar('Loss/Total', current_loss, global_step)
                writer.add_scalar('Loss/Reconstruction', loss_recon.item() / BATCH_SIZE, global_step)
                writer.add_scalar('Loss/KL', loss_kl.item() / BATCH_SIZE, global_step)

            # TensorBoard Immagini (Ogni 300 batch per vedere prima i risultati)
            if global_step % 300 == 0:
                with torch.no_grad():
                    model.eval()
                    comparison = torch.cat([x[:8], recon_x[:8]])
                    grid = vutils.make_grid(comparison, nrow=8, normalize=True)
                    writer.add_image('Visual_Reconstruction', grid, global_step)
                    model.train()

            global_step += 1
            epoch_loss += loss.item()

        # Salvataggio checkpoint epoca
        save_file = model_dir / "model.pth" 
        torch.save(model.state_dict(), str(save_file))
        
        avg_loss = epoch_loss / len(train_dataset)
        print(f"===> Epoch {epoch} completata. Loss Media: {avg_loss:.2f}")

    writer.close()
    print(f"Training completato. Modello salvato in: {model_dir}")

if __name__ == "__main__":
    train()