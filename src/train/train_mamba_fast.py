import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.mamba.mamba_vqvae import MambaVQVAE

# --- CONFIGURAZIONE VELOCE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKEN_FILE = "dataset/celeba_tokens.pt"
BATCH_SIZE = 480       # <--- ORA PUOI ALZARLO!
EPOCHS = 30
LR = 5e-4
VQ_DIM = 1024

def train():
    print(f"Caricamento dataset pre-calcolato da {TOKEN_FILE}...")
    if not os.path.exists(TOKEN_FILE):
        print("ERRORE: Devi prima eseguire 'src/dataset/prepare_tokens.py'!")
        return

    data = torch.load(TOKEN_FILE)
    indices_tensor = data['indices'].long() # Mamba vuole LongTensor
    attrs_tensor = data['attrs'].long()
    
    # Creiamo un dataset super leggero che sta tutto in RAM
    train_dataset = TensorDataset(indices_tensor, attrs_tensor)
    # num_workers=0 è più veloce qui perché i dati sono già in memoria RAM
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Dataset caricato: {len(train_dataset)} campioni.")

    # Modello
    model = MambaVQVAE(
        vocab_size=VQ_DIM,
        dim=512,
        state_size=16,
        num_layers=6
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    
    os.makedirs("checkpoints", exist_ok=True)

    print("Inizio training FAST...")
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_dl, leave=True)
        total_loss = 0
        
        for batch_idx, (indices, attrs) in enumerate(loop):
            indices = indices.to(DEVICE)
            attrs = attrs.to(DEVICE)
            
            with torch.amp.autocast("cuda"):
                # Forward Pass
                # Input: Indici [0...254] (escludiamo l'ultimo per input)
                input_seq = indices[:, :-1]
                target_seq = indices # Il target è l'intera sequenza corretta
                
                logits = model(input_seq, attrs) 
                
                # Logica Loss (Come prima):
                # Logits shape: [B, 3(Attr)+255(Img), Vocab]
                # Scartiamo i primi 3 token di output (corrispondenti alla predizione degli attributi successivi)
                # L'output all'indice 3 (dopo l'ultimo attributo) predice il pixel 0.
                img_logits = logits[:, 3:, :] # (B, 256, 1024)
                
                # Target: l'intera sequenza indici (B, 256)
                loss = criterion(img_logits.reshape(-1, VQ_DIM), target_seq.reshape(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Salvataggio ogni 5 epoche per risparmiare spazio, o sempre se vuoi
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/mamba_vq_fast_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()