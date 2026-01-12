import sys
import os

# Get the path to the 'src' folder and add it to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from torchvision.datasets import CelebA
from torchvision import transforms
import os
from torch.cuda.amp import GradScaler, autocast

# Import dai tuoi moduli
# Assicurati che config definisca le variabili, altrimenti usa quelle locali qui sotto
try:
    from config import *
except ImportError:
    pass

from model.mamba.mamba_image_model import MambaImageModel

# Parametri (se non definiti in config)
# NANO:
model_configs = {
    "nano": {
        "EMBEDDING_DIMENSION": 128,
        "STATE_SIZE": 8,
        "EXPANSION": 1.1,
        "CONV_KERNEL": 4,
        "MAMBA_LAYERS": 4
    },
    "mini": {
        "EMBEDDING_DIMENSION": 256,
        "STATE_SIZE": 12,
        "EXPANSION": 1.1,
        "CONV_KERNEL": 4,
        "MAMBA_LAYERS": 4
    },
    "base": {
        "EMBEDDING_DIMENSION": 512,
        "STATE_SIZE": 16,
        "EXPANSION": 1.1,
        "CONV_KERNEL": 4,
        "MAMBA_LAYERS": 5
    }
}
chosen = "nano"
EMBEDDING_DIMENSION = model_configs[chosen]["EMBEDDING_DIMENSION"]
STATE_SIZE = model_configs[chosen]["STATE_SIZE"]
EXPANSION = model_configs[chosen]["EXPANSION"]
CONV_KERNEL = model_configs[chosen]["CONV_KERNEL"]
MAMBA_LAYERS = model_configs[chosen]["MAMBA_LAYERS"]

BATCH_SIZE = 128  
LEARNING_RATE = 1e-4
EPOCHS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limitiamo il dataset per vedere se impara? (Opzionale)
USE_SUBSET = False          # Metti True se vuoi fare test veloci su poche immagini

def flatten_image(x):
    """
    Trasforma (1, 64, 64) -> (4096, 1)
    Deve essere definita qui fuori per funzionare con num_workers > 0 su Windows.
    """
    return x.view(-1, 1)

def save_checkpoint(model, optimizer, epoch, filename="nano_checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train():
    print(f"Avvio training con device: {DEVICE}")

    # Inizializza lo scaler per Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    # Creazione cartelle per checkpoint e log
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    # Inizializza TensorBoard
    writer = SummaryWriter("runs/mamba_celeba_experiment")

    # 1. Pipeline di Preprocessing
    preprocessing_pipeline = transforms.Compose([
        transforms.CenterCrop(178),                  
        transforms.Resize((64, 64)),                 
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),                        
        transforms.Lambda(flatten_image)
    ])

    # 2. Caricamento Dataset
    train_dataset = CelebA(
        root='./dataset',       
        split='train',       
        target_type='attr',  
        transform=preprocessing_pipeline,
        download=False       
    )

    if USE_SUBSET:
        n = 20_000
        print(f"⚠️ TRAINING SU SOTTOINSIEME RIDOTTO ({n} immagini) PER VELOCITÀ ⚠️")
        indices = torch.arange(n) # Usiamo solo le prime n immagini
        train_dataset = Subset(train_dataset, indices)

    # 3. DataLoader
    train_dl = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=5, # Velocizza il caricamento dati
        pin_memory=True
    )

    # 4. Modello
    model = MambaImageModel(
        dim=EMBEDDING_DIMENSION,
        state_size=STATE_SIZE,
        expansion=EXPANSION,
        conv_kernel=CONV_KERNEL,
        num_layers=MAMBA_LAYERS,
        num_attr=3
    ).to(DEVICE)

    # 5. Ottimizzatore
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    global_step = 0

    print("Inizio ciclo di training...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_dl, leave=True)
        total_loss = 0

        for batch_idx, (imgs, attrs) in enumerate(loop):
            # Preparazione dati
            imgs = imgs.to(DEVICE)  # (B, 4096, 1)
            # Selezioniamo 3 attributi: Male(20), Smiling(31), Young(39)
            selected_attrs = attrs[:, [20, 31, 39]].long().to(DEVICE) # (B, 3)
            
            # --- Forward Pass (Teacher Forcing) ---
            # --- Calcolo Loss ---
            # Target: pixels [1...4095] convertiti in indici interi 0-255
            # Nota: imgs è normalizzato [0,1], lo riportiamo a [0,255] per la CrossEntropy
            target_indices = (imgs.squeeze(-1) * 255).long() # (B, 4096)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Il modello riceve pixels [0...4094] e attributi
                logits = model(imgs, selected_attrs) # (B, 4096, 256)

                # Flattening per la loss: (B*4096, 256) vs (B*4096)
                loss = criterion(logits.reshape(-1, 256), target_indices.reshape(-1))

            # --- Backward Pass ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient Clipping (importante per Mamba/RNN per evitare esplosioni)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # --- Logging ---
            total_loss += loss.item()
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
            
            global_step += 1

        # --- Fine Epoca ---
        avg_loss = total_loss / len(train_dl)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        print(f"Epoch {epoch+1} completata. Avg Loss: {avg_loss:.4f}")

        # Salva Checkpoint
        save_path = f"checkpoints/nano_mamba_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch, save_path)

        # --- Generazione di Test (Visualizzazione) ---
        # Generiamo immagini solo ogni X epoche per risparmiare tempo
        if (epoch + 1) % 1 == 0: 
            print("Generazione immagini di test...")
            model.eval()
            with torch.no_grad():
                # Prendiamo gli attributi del primo batch per generare
                # Generiamo solo 4 immagini per non bloccare il training troppo a lungo
                sample_attrs = selected_attrs[:4] 
                
                # Questa funzione è lenta (4096 step sequenziali)
                generated_indices = model.generate(sample_attrs) # (4, 4096)
                
                # Reshape per visualizzazione: (B, 4096) -> (B, 1, 64, 64)
                gen_imgs = generated_indices.view(-1, 1, 64, 64).float() / 255.0
                
                # Creiamo una griglia
                grid = torchvision.utils.make_grid(gen_imgs, nrow=2)
                writer.add_image("Generated/Epoch_End", grid, epoch)
                
            model.train() # Rimetti in train mode

    writer.close()
    print("Training completato.")

if __name__ == "__main__":
    # Assicurati di aver corretto prediction_head.py prima di lanciare!
    train()


