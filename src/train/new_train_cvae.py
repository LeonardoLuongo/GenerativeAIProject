import os
import sys
import torch
import torch.nn as nn
import numpy as np
import io
import matplotlib.pyplot as plt
import PIL.Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_tensor
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURAZIONE ---
EXP_NAME = "cvae_final_VGG128sqrt_balanced_sl4s"
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
import config
from src.models.cvae import CelebACVAE
from src.models.perceptual_loss import VGGPerceptualLoss

# --- PARAMETRI ---
BATCH_SIZE = 128     
LATENT_DIM = 128    
LR = 2e-4            
EPOCHS = 80          
IMG_SIZE = 64
BETA = 1.0           
PERCEPTUAL_WEIGHT = 0.0015
ATTR_INDICES = [20, 31, 39] # Male, Smiling, Young
NUM_ATTR = len(ATTR_INDICES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SAMPLER  ---
def get_balanced_sampler(dataset, attr_indices):
    """WeightedRandomSampler"""
    # Portiamo attributi CelebA da [-1, 1] a [0, 1]
    all_attrs = (dataset.attr[:, attr_indices] + 1) // 2
    
    # Convertiamo le 8 combinazioni in indici 0-7
    multipliers = torch.tensor([4, 2, 1], dtype=torch.long)
    comb_indices = (all_attrs @ multipliers).numpy()
    
    # Contiamo le frequenze reali nel dataset
    counts = np.bincount(comb_indices, minlength=8)
    print(f"Frequenze originali classi (0-7): {counts}")
    
    '''
        Donna, Seria, Anziana	        3.499
        Donna, Seria, Giovane	        40.189
        Donna, Sorridente, Anziana	    7.668
        Donna, Sorridente, Giovane	    43.153
        Uomo, Serio, Anziano	        14.168
        Uomo, Serio, Giovane	        26.834
        Uomo, Sorridente, Anziano	    10.647
        Uomo, Sorridente, Giovane	    16.612
    '''

    # Il peso è l'inverso della frequenza
    weights_per_comb = 1.0 / np.sqrt(counts)
    sample_weights = torch.from_numpy(weights_per_comb[comb_indices])
    
    return WeightedRandomSampler(sample_weights, len(dataset), replacement=True)


def plot_classes_to_tensorboard(model, device, latent_dim):
    model.eval()
    scenari = [
        {"desc": "F-SER-ANZ", "attr": [0., 0., 0.]}, {"desc": "F-SER-GIO", "attr": [0., 0., 1.]},
        {"desc": "F-SOR-ANZ", "attr": [0., 1., 0.]}, {"desc": "F-SOR-GIO", "attr": [0., 1., 1.]},
        {"desc": "M-SER-ANZ", "attr": [1., 0., 0.]}, {"desc": "M-SER-GIO", "attr": [1., 0., 1.]},
        {"desc": "M-SOR-ANZ", "attr": [1., 1., 0.]}, {"desc": "M-SOR-GIO", "attr": [1., 1., 1.]},
    ]
    num_variants = 4
    fig, axes = plt.subplots(len(scenari), num_variants, figsize=(10, 14))
    
    with torch.no_grad():
        for i, scenario in enumerate(scenari):
            cond = torch.tensor([scenario["attr"]]).to(device).float()
            cond_expanded = cond.expand(num_variants, -1)
            z = torch.randn(num_variants, latent_dim).to(device) # Temperatura 1.0 per monitoraggio
            z_cond = torch.cat((z, cond_expanded), dim=1)
            
            # Decoder
            y = model.dec_input(z_cond).view(-1, 512, 4, 4)
            y = model.up1(model._inject_cond(y, cond_expanded))
            y = model.up2(model._inject_cond(y, cond_expanded))
            y = model.up3(model._inject_cond(y, cond_expanded))
            y = model.up4(model._inject_cond(y, cond_expanded))
            
            for j in range(num_variants):
                img = y[j].cpu().permute(1, 2, 0).numpy()
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0: axes[i, j].set_title(scenario["desc"], fontsize=9, loc='left', fontweight='bold')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return to_tensor(PIL.Image.open(buf))

# --- FUNZIONI LOSS ---
reconstruction_loss_function = nn.BCELoss(reduction='sum')

def kl_loss_function(mu, log_sigma):
    """KL Divergence (Somma)"""
    return 0.5 * torch.sum(mu**2 + torch.exp(2 * log_sigma) - 1 - 2 * log_sigma)

# --- TRAINING LOOP ---
def train():
    model_dir = config.MODEL_PATH / EXP_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.TRAIN_RES_PATH / EXP_NAME)

    transform = transforms.Compose([
        transforms.CenterCrop(178),           
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),               
    ])

    train_dataset = datasets.CelebA(root=str(config.DATASET_BASIC_PATH), split='train', target_type='attr', transform=transform, download=False)
    
    # Bilanciamento Dataset
    sampler = get_balanced_sampler(train_dataset, ATTR_INDICES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    # Inizializzazione
    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Dimezza il LR ogni 20 epoche per i dettagli finali
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"--- START TRAINING: {EXP_NAME} ---")
    
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, attr in progress_bar:
            # Normalizzazione attributi CelebA [-1, 1] -> [0, 1]
            cond = ((attr[:, ATTR_INDICES].float() + 1) / 2).to(DEVICE)
            x = x.to(DEVICE)

            optimizer.zero_grad()
            output, mu, log_sigma = model(x, cond)
            
            # Calcolo Loss
            recon_loss = reconstruction_loss_function(output, x)
            kl_loss = kl_loss_function(mu, log_sigma)
            vgg_loss = perceptual_loss_fn(output, x)
            
            # Loss Totale 
            loss = recon_loss + (BETA * kl_loss) + (PERCEPTUAL_WEIGHT * vgg_loss)
            
            loss.backward()
            # Gradient clipping per i Residual Blocks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.0f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})

            # TensorBoard Scalars
            if global_step % 10 == 0:
                writer.add_scalar('Loss/BCE_Sum', recon_loss.item(), global_step)
                writer.add_scalar('Loss/KL_Sum', kl_loss.item(), global_step)
                writer.add_scalar('Loss/VGG_Weighted', (PERCEPTUAL_WEIGHT * vgg_loss).item(), global_step)
                writer.add_scalar('Loss/Total', loss.item(), global_step)

            # TensorBoard Images
            if global_step % 500 == 0:
                with torch.no_grad():
                    model.eval()
                    comparison = torch.cat([x[:8], output[:8]])
                    grid = vutils.make_grid(comparison, nrow=8, normalize=True)
                    writer.add_image('Visual/Reconstruction', grid, global_step)
                    model.train()

            if global_step % 1000 == 0:
                with torch.no_grad():
                    model.eval()
                    labeled_grid = plot_classes_to_tensorboard(model, DEVICE, LATENT_DIM)
                    writer.add_image('Visual/All_8_Classes_Labled', labeled_grid, global_step)
                    writer.flush()
                    model.train()
            
            global_step += 1

        scheduler.step()
        print(f"===> Epoch {epoch} terminata. Loss Media: {total_epoch_loss / len(train_loader):.0f}")
        torch.save(model.state_dict(), str(model_dir / "model.pth"))

    writer.close()
    print(f"Training concluso. Modello salvato in {model_dir}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    train()