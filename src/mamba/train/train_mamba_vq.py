import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_dir)

# --- ASSICURATI CHE QUESTE IMPORTAZIONI SIANO CORRETTE ---
from vqvae.model.vq_vae import VQVAE
from mamba.model.mamba_vqvae import MambaVQVAE

# --- CONFIGURAZIONE ---
MAMBA_PATH = None 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH_VQ = "checkpoints/vqvae_16x16_2_epoch_50.pth"
BATCH_SIZE = 96 # Riduco leggermente per sicurezza memoria
EPOCHS = 100
LR = 0.00025
VQ_DIM = 1024  
ATTR_NAMES = ['Male', 'Smiling', 'Young']

# --- FUNZIONI DI UTILITÀ ---

def calculate_codebook_metrics(indices, num_embeddings):
    """
    Calcola metriche sulla diversità dei token generati.
    Serve a capire se il modello sta usando tutto il codebook o è in 'mode collapse'.
    """
    indices = indices.flatten().float()
    
    # Conta quante volte ogni codice viene usato
    counts = torch.histc(indices, bins=num_embeddings, min=0, max=num_embeddings - 1)
    
    # 1. Metrica: Quanti codici unici sono stati usati in questo batch?
    # Se è molto basso (es. < 10 su 1024), il modello genera immagini piatte/uguali.
    used_indices = (counts > 0).sum().item()
    unique_ratio = used_indices / num_embeddings

    # 2. Metrica: Perplessità (Entropia esponenziale)
    # Indica quanto è uniforme la distribuzione. Più alta = più entropia = più varietà (di solito bene).
    probs = counts / counts.sum()
    # Evitiamo log(0)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))
    perplexity = torch.exp(entropy).item()

    return unique_ratio, perplexity

def create_grid_figure(images_tensor, attrs_tensor, attr_names):
    """
    Crea una figura Matplotlib con la griglia delle 8 immagini e i titoli.
    images_tensor: (8, 3, 64, 64)
    attrs_tensor: (8, 3)
    """
    # Denormalizza immagini da [-1, 1] a [0, 1] per la visualizzazione
    images_tensor = images_tensor.cpu() * 0.5 + 0.5
    images_tensor = torch.clamp(images_tensor, 0, 1)
    
    num_imgs = images_tensor.shape[0]
    cols = 4
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    axes = axes.flatten()
    
    for i in range(num_imgs):
        ax = axes[i]
        # Immagine
        img_np = images_tensor[i].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.axis('off')
        
        # Titolo basato sugli attributi
        current_attrs = attrs_tensor[i].tolist()
        title_parts = []
        for name, val in zip(attr_names, current_attrs):
            if val == 1:
                title_parts.append(name)
            else:
                title_parts.append(f"Not {name}")
        
        # Esempio titolo: "Male, Not Smiling, Young"
        title_str = "\n".join(title_parts)
        ax.set_title(title_str, fontsize=10, color='black')

    plt.tight_layout()
    
    # Salva su buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
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
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_fixed_attributes_batch():
    """
    Genera le 8 combinazioni possibili per 3 attributi binari.
    Ord: 000, 001, 010, 011, 100, 101, 110, 111
    """
    # Male(20), Smiling(31), Young(39)
    attrs = []
    for m in [0, 1]:
        for s in [0, 1]:
            for y in [0, 1]:
                attrs.append([m, s, y])
    return torch.tensor(attrs, dtype=torch.long).to(DEVICE)

# --- TRAINING LOOP ---

def train():
    # 1. Dataset
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CelebA(root='./dataset', split='train', target_type='attr', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Modelli
    vqvae = load_vqvae()
    
    mamba = MambaVQVAE(
        vocab_size=VQ_DIM,
        dim=512,
        state_size=16,
        num_layers=16,
        expansion=2.0
    ).to(DEVICE)

    if MAMBA_PATH is not None and os.path.exists(MAMBA_PATH):
        print(f"Caricamento Mamba da {MAMBA_PATH}")
        mamba.load_state_dict(torch.load(MAMBA_PATH, map_location=DEVICE))
    else:
        print("Addestramento Mamba da zero.")

    optimizer = torch.optim.AdamW(mamba.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="runs/mamba_vqvae_grid")
    scaler = torch.amp.GradScaler("cuda") 
    
    # Prepara il batch fisso di 8 attributi per la validazione visiva
    fixed_attrs = get_fixed_attributes_batch() # Shape (8, 3)

    print(f"Inizio training ({DEVICE})...")

    global_step = 0
    for epoch in range(EPOCHS):
        mamba.train()
        loop = tqdm(dataloader, leave=True)
        total_loss = 0

        # Variabili per accumulare metriche durante l'epoca
        epoch_perplexity = 0
        epoch_unique_codes = 0
        num_batches = 0

        for imgs, attrs in loop:
            imgs = imgs.to(DEVICE)
            # Indici CelebA: Male(20), Smiling(31), Young(39)
            attrs = attrs[:, [20, 31, 39]].to(DEVICE)

            with torch.no_grad():
                # Indices shape: (B, 256) per immagini 64x64 con VQVAE 16x16
                indices = vqvae.get_codebook_indices(imgs)

            with torch.amp.autocast("cuda"):
                # Forward: Input completo
                logits = mamba(indices, attrs) # Output 260 elementi
                
                # --- SLICING CRITICO (Sequenziale) ---
                # Index 0 (SOS) -> Predice Attr1
                # ...
                # Index 3 (Attr3) -> Predice P0 (Primo Pixel)
                # ...
                # Index 258 (P254) -> Predice P255 (Ultimo Pixel)
                
                # Prendiamo da 3 a 258 (escluso 259)
                img_logits = logits[:, 3:-1, :] 
                
                # Target: P0 ... P255
                target_seq = indices 
                
                loss = criterion(img_logits.reshape(-1, VQ_DIM), target_seq.reshape(-1))

            optimizer.zero_grad()

            # Invece di loss.backward():
            scaler.scale(loss).backward()

            # Unscale prima del clipping (importante!)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)

            # Invece di optimizer.step():
            scaler.step(optimizer)
            scaler.update()


            total_loss += loss.item()
            
            # --- Calcolo Metriche Training (su un sottoinsieme per velocità) ---
            if global_step % 50 == 0:
                with torch.no_grad():
                    # Prendiamo gli argmax dei logits per vedere cosa avrebbe predetto il modello
                    pred_indices = torch.argmax(logits, dim=-1)
                    u_ratio, perp = calculate_codebook_metrics(pred_indices, VQ_DIM)
                    
                    writer.add_scalar('Metrics/Train_Unique_Ratio', u_ratio, global_step)
                    writer.add_scalar('Metrics/Train_Perplexity', perp, global_step)
            
            # current_lr = scheduler.get_last_lr()[0]
            loop.set_postfix(loss=loss.item(), perp=f"{perp:.1f}")
            
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Epoch', avg_loss, epoch+1)

        # --- VALIDAZIONE / GENERAZIONE ---
        print("Generazione Griglia 8 Combinazioni...")
        mamba.eval()
        with torch.no_grad():
            # 1. Generazione Indici
            # Passiamo (8, 3) tensore di attributi
            generated_indices = mamba.generate(fixed_attrs, steps=256, temperature=1.0) # (8, 256)
            
            # 2. Metriche sui dati generati (fondamentale per capire se sta collassando)
            gen_u_ratio, gen_perp = calculate_codebook_metrics(generated_indices, VQ_DIM)
            print(f" >> Generated Metrics | Unique Ratio: {gen_u_ratio:.3f} | Perplexity: {gen_perp:.2f}")
            writer.add_scalar('Metrics/Generated_Unique_Ratio', gen_u_ratio, epoch+1)
            writer.add_scalar('Metrics/Generated_Perplexity', gen_perp, epoch+1)

            # 3. Decodifica Immagini
            generated_img = vqvae.decode_from_indices(generated_indices) # (8, 3, 64, 64)
            
            # 4. Creazione Griglia con Titoli
            grid_tensor = create_grid_figure(generated_img, fixed_attrs, ATTR_NAMES)
            
            writer.add_image('Generated/8_Attributes_Grid', grid_tensor, epoch+1)

        # Salvataggio
        if (epoch + 1) % 5 == 0:
            torch.save(mamba.state_dict(), f"checkpoints/mamba_vq_epoch_{epoch+1}.pth")
            print(f"Checkpoint salvato: epoch {epoch+1}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    train()