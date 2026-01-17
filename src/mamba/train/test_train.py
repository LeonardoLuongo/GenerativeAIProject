import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms.functional as F

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = (os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(src_dir)

from vqvae.model.vq_vae import VQVAE
from mamba.model.mamba_vqvae import MambaVQVAE

# --- CONFIGURAZIONE DEBUG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH_VQ = "checkpoints/vqvae_16x16_2_epoch_50.pth"
BATCH_SIZE = 16    # Piccolo per overfit
LR = 1e-4          # Basso e costante per stabilità
VQ_DIM = 1024

def plot_img_with_labels(img_tensor, attrs_tensor, attr_names):
    np_img = img_tensor.permute(1, 2, 0).numpy()
    labels_text = "Attr: "
    current_attrs = attrs_tensor[0].cpu().tolist()
    for name, val in zip(attr_names, current_attrs):
        if val == 1: labels_text += f"{name} "
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np_img)
    ax.axis('off')
    ax.set_title(labels_text)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return F.to_tensor(image)

def load_vqvae():
    print("Caricamento VQ-VAE...")
    model = VQVAE(num_embeddings=VQ_DIM, embedding_dim=256).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH_VQ):
        raise FileNotFoundError(f"Checkpoint VQ-VAE non trovato in {CHECKPOINT_PATH_VQ}")
    
    ckpt = torch.load(CHECKPOINT_PATH_VQ, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

# --- AGGIUNGI QUESTA FUNZIONE DI PLOT ---
def plot_comparison(gen_tensor, truth_tensor, title):
    """Confronta Generazione Mamba vs Verità VQ-VAE"""
    gen_np = gen_tensor.permute(1, 2, 0).cpu().numpy()
    truth_np = truth_tensor.permute(1, 2, 0).cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    ax1.imshow(gen_np)
    ax1.set_title("MAMBA (Gen)")
    ax1.axis('off')
    
    ax2.imshow(truth_np)
    ax2.set_title("VQ-VAE (Target)")
    ax2.axis('off')
    
    plt.suptitle(title)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return F.to_tensor(image)

def train_debug():
    # 1. Dataset
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CelebA(root='./dataset', split='train', target_type='attr', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Modelli
    vqvae = load_vqvae()
    # Usa un modello piccolo per fare presto
    mamba = MambaVQVAE(vocab_size=VQ_DIM, dim=512, state_size=16, num_layers=4).to(DEVICE)

    optimizer = torch.optim.AdamW(mamba.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    writer = SummaryWriter(log_dir="runs/mamba_debug")

    # 3. FIX BATCH PER OVERFITTING
    print("Estrazione Fixed Batch...")
    fixed_imgs, fixed_raw_attrs = next(iter(dataloader))
    fixed_imgs = fixed_imgs.to(DEVICE)
    fixed_attrs = fixed_raw_attrs[:, [20, 31, 39]].to(DEVICE)
    
    # Target Ground Truth (Cosa dovrebbe uscire se Mamba fosse perfetto)
    with torch.no_grad():
        fixed_indices = vqvae.get_codebook_indices(fixed_imgs) # (B, 256)
        # Ricostruiamo subito l'immagine target dagli indici per il confronto
        target_img_recon = vqvae.decode_from_indices(fixed_indices)

    print("Inizio Overfitting Loop (Target Loss < 0.05)...")
    
    mamba.train()
    pbar = tqdm(range(1001)) # 1000 step
    
    for step in pbar:
        with torch.amp.autocast("cuda"):
            input_indices = fixed_indices[:, :-1]
            logits = mamba(input_indices, fixed_attrs) 
            img_logits = logits[:, 4:, :] 
            target_seq = fixed_indices[:, :-1]
            loss = criterion(img_logits.reshape(-1, VQ_DIM), target_seq.reshape(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        writer.add_scalar('Loss/Overfit', loss.item(), step)

        # Log ogni 100 step
        if step % 100 == 0:
            mamba.eval()
            with torch.no_grad():
                # --- GENERAZIONE ARGMAX (NO TEMPERATURA) ---
                # Usiamo temperatura bassissima per simulare argmax
                test_attr = fixed_attrs[0:1] 
                gen_idx = mamba.generate(test_attr, steps=256, temperature=1e-5) 
                
                gen_img = vqvae.decode_from_indices(gen_idx)
                
                # Prepara tensori per plot (denormalizza)
                g_t = torch.clamp(gen_img.squeeze(0).cpu() * 0.5 + 0.5, 0, 1)
                t_t = torch.clamp(target_img_recon[0].cpu() * 0.5 + 0.5, 0, 1) # Immagine Target
                
                final_img = plot_comparison(g_t, t_t, f"Step {step} | Loss: {loss.item():.4f}")
                writer.add_image('Debug/Compare', final_img, step)
            mamba.train()
            
if __name__ == "__main__":
    train_debug()