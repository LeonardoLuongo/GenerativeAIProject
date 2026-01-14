import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.vq_vae.vq_vae import VQVAE
from model.mamba.mamba_vqvae import MambaVQVAE


import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms.functional as F

def plot_img_with_labels(img_tensor, attrs_tensor, attr_names):
    """
    img_tensor: tensore (3, H, W) normalizzato tra 0 e 1
    attrs_tensor: tensore (1, 3) con valori 0 o 1
    attr_names: lista di stringhe es. ['Male', 'Smiling', 'Young']
    """
    # Converti immagine per matplotlib (C, H, W) -> (H, W, C)
    np_img = img_tensor.permute(1, 2, 0).numpy()
    
    # Prepara il testo degli attributi
    # attrs_tensor[0] perché il batch è 1
    labels_text = "Input Attributes:\n"
    labels_text += "-" * 20 + "\n"
    
    current_attrs = attrs_tensor[0].cpu().tolist()
    
    for name, val in zip(attr_names, current_attrs):
        status = "YES" if val == 1 else "NO"
        # Colore verde se presente, rosso se assente (gestito visivamente nel testo)
        labels_text += f"{name}: {status}\n"

    # Crea la figura (H, W) in pollici
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Mostra immagine
    ax1.imshow(np_img)
    ax1.axis('off')
    ax1.set_title("Generated Image")
    
    # Mostra testo
    ax2.text(0.1, 0.5, labels_text, fontsize=14, va='center', fontfamily='monospace')
    ax2.axis('off')
    
    # Salva la figura in un buffer di memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
     
    # Converti buffer in Tensore
    image = Image.open(buf)
    image_tensor = F.to_tensor(image)
    return image_tensor



# --- CONFIGURAZIONE ---
MAMBA_PATH = "checkpoints/mamba_vq_epoch_85.pth"  # Percorso checkpoint Mamba (se esiste)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH_VQ = "checkpoints/vqvae_16x16_2_epoch_50.pth"
BATCH_SIZE = 384
EPOCHS = 500
LR = 5e-4
VQ_DIM = 1024  # Codebook size

def load_vqvae():
    print("Caricamento VQ-VAE...")
    model = VQVAE(num_embeddings=VQ_DIM, embedding_dim=256).to(DEVICE)
    # Carica pesi
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

def train():
    # 1. Dataset e Dataloader
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CelebA(root='./dataset', split='train', target_type='attr', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Modelli
    vqvae = load_vqvae() # Modello congelato
    
    mamba = MambaVQVAE(
        vocab_size=VQ_DIM,
        dim=512,
        state_size=16, # Parametri del prof
        num_layers=6
    ).to(DEVICE)
    if MAMBA_PATH is not None and os.path.exists(MAMBA_PATH):
        mamba.load_state_dict(torch.load(MAMBA_PATH, map_location=DEVICE))
    else:
        print("Nessun checkpoint Mamba trovato, addestramento da zero.")

    optimizer = torch.optim.AdamW(mamba.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    # Questo scheduler parte basso, sale veloce fino a max_lr, e poi scende a zero.
    # È famoso per far convergere i modelli in meno epoche.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,              # Spingiamo un po' di più il LR (prima era 5e-4)
        epochs=EPOCHS,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,            # Il 10% del tempo lo usa per salire (warmup)
        div_factor=10,            # Parte da max_lr / 10
        final_div_factor=100      # Finisce a max_lr / 100
    )

    writer = SummaryWriter(log_dir="runs/mamba_vqvae")

    print(f"Inizio training Mamba su Latent Space ({DEVICE})...")

    for epoch in range(EPOCHS):
        mamba.train()
        loop = tqdm(dataloader, leave=True)
        total_loss = 0

        for imgs, attrs in loop:
            imgs = imgs.to(DEVICE)
            # Seleziona attributi: Male(20), Smiling(31), Young(39)
            attrs = attrs[:, [20, 31, 39]].to(DEVICE)

            with torch.no_grad():
                # Otteniamo gli indici discreti dal VQ-VAE (B, 256)
                # Il VQ-VAE 16x16 produce 256 token
                indices = vqvae.get_codebook_indices(imgs)

            with torch.amp.autocast("cuda"):
                # Forward Pass
                # Passiamo l'intera sequenza di indici. Il modello gestirà internamente 
                # il teacher forcing shiftando gli input.
                
                # Input sequenza: [SOS, Attrs, Idx_0 ... Idx_254] (Lunghezza total 1+3+256)
                # Tuttavia, nel forward pass gli passiamo (B, 256) indici.
                # L'ultimo indice (255) non serve come input perché non deve predire nulla dopo.
                input_indices = indices[:, :-1] # (B, 255)
                
                logits = mamba(input_indices, attrs) # Output: (B, 4 + 255, Vocab)
                
                # --- CALCOLO LOSS ---
                # Il modello ha prodotto logits per tutta la sequenza: [SOS, A1, A2, A3, P0...P254]
                # Noi vogliamo che:
                # - L'output allo step A3 predica P0 (indice 0 dell'immagine)
                # - L'output allo step P(i) predica P(i+1)
                
                # Tagliamo i primi 4 output (relativi a SOS e Attributi che predicono il primo pixel)
                # I logits utili partono dall'indice 3 (che corrisponde all'ultimo attributo processato)
                # fino alla fine.
                
                # Logits shape: (B, 259, 1024) -> [SOS, A1, A2, A3, P0...P254]
                # Vogliamo predire: [P0, P1, ... P255]
                # Aspetta, se l'input è lungo 255, l'output immagine è lungo 255.
                # Manca l'ultimo token target P255.
                
                # Rettifica logica forward:
                # Mamba output length = 1 (SOS) + 3 (Attr) + 255 (Input Pixels) = 259 steps.
                # Step 0 (SOS) -> Predice A1 (ignoriamo)
                # Step 1 (A1)  -> Predice A2 (ignoriamo)
                # Step 2 (A2)  -> Predice A3 (ignoriamo)
                # Step 3 (A3)  -> Predice P0 (Questo ci serve!)
                # ...
                # Step 258 (P254)-> Predice P255 (Questo ci serve!)
                
                # Quindi prendiamo logits[:, 3:, :] -> Shape (B, 256, 1024)
                img_logits = logits[:, 3:, :]
                
                loss = criterion(img_logits.reshape(-1, VQ_DIM), indices.reshape(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            current_lr = scheduler.get_last_lr()[0]
            loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.6f}")


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        print("Generazione in corso (256 step)...")
        mamba.eval()
        attrs = torch.randint(0, 2, (1, 3)).to(DEVICE)
        with torch.no_grad():
            
            # 1. Mamba genera gli indici (token)
            generated_indices = mamba.generate(attrs, steps=256) # (1, 256)
            
            # 2. VQ-VAE decodifica gli indici in immagine
            # decode_from_indices si aspetta (B, H, W) o (B, SeqLen) gestito internamente
            generated_img = vqvae.decode_from_indices(generated_indices)
        
        img_tensor = generated_img.squeeze(0).cpu() # (3, 64, 64)
        img_tensor = img_tensor * 0.5 + 0.5         # Denormalize
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Definisci i nomi degli attributi che stai usando (indici 20, 31, 39)
        my_attr_names = ['Male', 'Smiling', 'Young']

        # Crea l'immagine combinata (Immagine + Testo)
        # Passiamo attrs (che è il tensore di input originale)
        combined_image = plot_img_with_labels(img_tensor, attrs, my_attr_names)

        writer.add_scalar('Loss/Train', avg_loss, epoch+1)

        # Logga l'immagine combinata invece di quella raw
        writer.add_image(f'Generated_with_Labels/Epoch_{epoch+1}', combined_image, epoch+1)

        # Salvataggio
        if (epoch + 1) % 5 == 0:
            torch.save(mamba.state_dict(), f"checkpoints/mamba_vq_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()