import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from torchvision.datasets import CelebA
from torchvision import transforms

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.mamba.mamba_patch_model import PatchMambaModel

# --- SOBEL LOSS (Invariata) ---
class SobelLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
        self.sobel_x = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

# --- CONFIG ---
# Riduciamo leggermente il batch size se necessario perché la sequenza è 4x più lunga (256 tokens)
BATCH_SIZE = 64 
LR = 1e-4
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"--- PATCH MAMBA RGB 4x4 (Noise Injected) ---")
    
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(), 
    ])

    train_dataset = CelebA(root='./dataset', split='train', target_type='attr', transform=transform, download=False)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = PatchMambaModel(
        dim=128,         
        state_size=32,   # Stato aumentato per gestire sequenza più lunga (256 passi)
        num_layers=6,
        patch_size=4,    # <--- MODIFICA CHIAVE: PATCH PICCOLA PER I DETTAGLI
        channels=3,
        latent_dim=128   # <--- MODIFICA CHIAVE: DIMENSIONE RUMORE
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    loss_mse = nn.MSELoss()  
    loss_l1 = nn.L1Loss()    
    loss_sobel = SobelLoss(DEVICE) 
    
    writer = SummaryWriter("runs/patch_mamba_rgb_noise")
    global_step = 0

    _, fixed_attrs = next(iter(dataloader))
    fixed_attrs = fixed_attrs[:, [20, 31, 39]].to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader)
        
        for batch_imgs, batch_attrs in loop:
            batch_imgs = batch_imgs.to(DEVICE)
            attrs = batch_attrs[:, [20, 31, 39]].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward (ora include la generazione interna di rumore)
            predicted_patches, refined_image = model(batch_imgs, attrs)
            
            with torch.no_grad():
                target_patches = model.patchify(batch_imgs) 
            
            # Loss Ibrida
            l_mamba_mse = loss_mse(predicted_patches, target_patches)
            l_mamba_l1  = loss_l1(predicted_patches, target_patches)
            loss_mamba = l_mamba_mse + 0.5 * l_mamba_l1
            
            l_img_l1 = loss_l1(refined_image, batch_imgs)
            l_img_sobel = loss_sobel(refined_image, batch_imgs)
            
            # Loss Totale
            total_loss = loss_mamba + l_img_l1 + 0.7 * l_img_sobel 
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loop.set_postfix(loss=total_loss.item())
            writer.add_scalar("Loss/Total", total_loss.item(), global_step)
            global_step += 1
            
        # Checkpoint e Preview
        if (epoch+1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                # Nota: generate() ora creerà rumore casuale ogni volta
                gen_imgs = model.generate(fixed_attrs[:8])
                grid = torchvision.utils.make_grid(gen_imgs, nrow=4)
                writer.add_image("Generated/RGB_Noise", grid, epoch)
                torch.save(model.state_dict(), f"checkpoints/patch_mamba_rgb_noise_ep{epoch+1}.pth")

if __name__ == "__main__":
    train()