import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from torchvision.datasets import CelebA
from torchvision import transforms
import os

# Import dai tuoi moduli
# Assicurati che config definisca le variabili, altrimenti usa quelle locali qui sotto
try:
    from config import *
except ImportError:
    pass

from model.mamba.mamba_image_model import MambaImageModel

# Parametri (se non definiti in config)
EMBEDDING_DIMENSION = 256 
STATE_SIZE = 16
EXPANSION = 2
CONV_KERNEL = 4
MAMBA_LAYERS = 8
BATCH_SIZE = 32  
LEARNING_RATE = 1e-4
EPOCHS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_image(x):
    """
    Trasforma (1, 64, 64) -> (4096, 1)
    Deve essere definita qui fuori per funzionare con num_workers > 0 su Windows.
    """
    return x.view(-1, 1)

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train():
    print(f"Avvio training con device: {DEVICE}")
    
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

    # 3. DataLoader
    train_dl = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, # Velocizza il caricamento dati
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
            # Selezioniamo 3 attributi: Male(20), Smiling(31), Young(39) (esempio)
            selected_attrs = attrs[:, [20, 31, 39]].long().to(DEVICE) # (B, 3)
            
            # --- Forward Pass (Teacher Forcing) ---
            # Il modello riceve pixels [0...4094] e attributi
            logits = model(imgs, selected_attrs) # (B, 4096, 256)

            # --- Calcolo Loss ---
            # Target: pixels [1...4095] convertiti in indici interi 0-255
            # Nota: imgs è normalizzato [0,1], lo riportiamo a [0,255] per la CrossEntropy
            target_indices = (imgs.squeeze(-1) * 255).long() # (B, 4096)
            
            # Flattening per la loss: (B*4096, 256) vs (B*4096)
            loss = criterion(logits.reshape(-1, 256), target_indices.reshape(-1))

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (importante per Mamba/RNN per evitare esplosioni)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

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
        save_path = f"checkpoints/mamba_epoch_{epoch+1}.pth"
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


