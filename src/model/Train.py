import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import os
from config import *
from Network import ConditionalNetwork
from NoiseSchedule import NoiseSchedule
from Dataset import CelebADataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parametri del Modello e Training
L = 1000                # Step di diffusione
IMAGE_DIMENSIONS = (3, 64, 64)
BATCH_SIZE = 256
LEARNING_RATE = 1e-4    # 0.0001 è standard per Adam
WARMUP_EPOCHS = 5
EPOCHS = 60

# Percorsi
IMG_DIR = DATASET_CELEBA_PROCESSED_DATA_64_64
ATTR_FILE = DATASET_CELEBA_PROCESSED_DATA_LIST_ATTR
CHECKPOINT_DIR = './checkpoints'

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# Creiamo la cartella per salvare i modelli se non esiste
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

print(f"Inizializzazione su device: {device}")

# Dataset e Dataloader
print("Caricamento dataset...")
dataset = CelebADataset(IMG_DIR, ATTR_FILE)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

noise_schedule = NoiseSchedule(L, device=device)

# Creazione Modello
print("Creazione modello...")
# Passiamo L al costruttore come abbiamo modificato prima
model = ConditionalNetwork(L=L)
model = model.to(device)

ema = EMA(model, decay=0.999)

# Optimizer e Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Variabile globale per contare le epoche
epoch_count = 0

def get_lr(epoch):
    """Funzione semplice per warmup lineare"""
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    return LEARNING_RATE

def save_sample_images(epoch, model, noise_schedule, device):
    """
    Generazione immagini usando DDIM (Deterministico).
    Produce risultati più nitidi e coerenti rispetto a DDPM.
    """
    # Carica pesi mediati
    ema.apply_shadow()
    model.eval()
    with torch.no_grad():
        n_samples = 8
        
        # 1. Rumore Iniziale (Latent Code)
        # In DDIM, se fissiamo questo x iniziale, otterremo sempre la stessa faccia identica.
        x = torch.randn(n_samples, 3, 64, 64).to(device)
        
        # Attributi di test
        attrs = torch.zeros(n_samples, 3).to(device)
        attrs[:4] = torch.tensor([1, 1, 1]) # Uomo, Sorriso, Giovane
        attrs[4:] = torch.tensor([0, 0, 0]) # Donna, No Sorriso, Giovane
        
        # 2. Ciclo di Sampling DDIM
        for i in range(L - 1, -1, -1):
            t = torch.tensor([i] * n_samples, device=device)
            
            # A. Predizione del rumore
            eps = model(x, t, attrs)
            
            # B. Recupero Alpha Cumulativi (alpha_bar)
            # noise_schedule.alpha contiene già il prodotto cumulativo (quello che i paper chiamano alpha_bar)
            alpha_bar_t = noise_schedule.alpha[i]
            
            # Gestione del passo finale (t=0 -> t=-1)
            # Se siamo a t=0, il passo precedente è "nessun rumore", quindi alpha_bar = 1.0
            if i > 0:
                alpha_bar_prev = noise_schedule.alpha[i - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0).to(device)

            # C. Formula DDIM
            # 1. Stimiamo l'immagine pulita originale (x0) partendo da xt e dal rumore predetto
            #    x0_pred = (xt - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
            # (Opzionale ma consigliato) Clampiamo x0 per stabilità
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # 2. Calcoliamo la "direzione" che punta verso il rumore al passo t-1
            #    dir_xt = sqrt(1 - alpha_bar_prev) * eps
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            
            # 3. Calcoliamo il nuovo x_{t-1}
            #    x_{t-1} = sqrt(alpha_bar_prev) * x0_pred + dir_xt
            #    (Nota: Non c'è il termine "+ sigma * z", per questo è deterministico)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        # 3. Salvataggio
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1) # Rimuove pixel bruciati
        
        save_path = os.path.join(CHECKPOINT_DIR, f"sample_epoch_{epoch}_DDIM.png")
        vutils.save_image(x, save_path, nrow=4)
        print(f"   -> Generata anteprima DDIM: {save_path}")
    
    model.train()
    ema.restore() #Ripristina pesi originali per il training

def training_epoch(dataloader):
    global epoch_count
    model.train() # Mette il modello in modalità training (abilita dropout/batchnorm)
    average_loss = 0.0
    
    current_lr = get_lr(epoch_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
        
    print(f"Epoch {epoch_count} start. LR: {current_lr:.6f}")

    # Barra di progresso testuale
    total_steps = len(dataloader)
    
    for step, (x, attributes) in enumerate(dataloader):
        # Spostiamo i dati sulla GPU
        x = x.to(device)
        attributes = attributes.to(device)
        
        n = x.shape[0] # Minibatch size

        # Genera indici di tempo casuali
        t = torch.randint(0, L, (n,), device=device)

        # Genera rumore casuale (Target)
        eps = torch.randn_like(x)

        # Calcola l'immagine rumorosa (Input)
        # Recuperiamo i coefficienti alpha dal noise_schedule
        sqrt_alpha = noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        sqrt_1_alpha = noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)
        
        # Forward Diffusion
        zt = sqrt_alpha * x + sqrt_1_alpha * eps

        # Predizione della rete
        # Passiamo zt (immagine), t (tempo), attributes (condizione)
        g = model(zt, t, attributes)
        loss = F.mse_loss(g,eps)

       # Protezione NaN
        if torch.isnan(loss):
            print(f"!!! ATTENZIONE: LOSS NAN allo step {step}. Salto il batch.")
            optimizer.zero_grad() 
            continue

        optimizer.zero_grad() # Azzera i gradienti vecchi
        loss.backward()       # Calcola i nuovi gradienti

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #per evitare che i gradienti alti generino pesi infiniti o nan
        optimizer.step()      # Aggiorna i pesi
        ema.update()

        # Aggiornamento media mobile della loss (per il print)
        average_loss = 0.9 * average_loss + 0.1 * loss.cpu().item()
        
        # Print ogni 100 step per vedere che succede
        if step % 100 == 0:
            print(f"   Step {step}/{total_steps} | Loss corrente: {loss.item():.4f}")

    epoch_count += 1
    print(f'>>> Epoch {epoch_count} completata. Average loss={average_loss:.4f}')
    
    # SALVATAGGIO CHECKPOINT
    # Fondamentale: salviamo il modello alla fine di ogni epoca
    save_path = os.path.join(CHECKPOINT_DIR, f"ddpm_celebA_ep{epoch_count}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in: {save_path}")
    save_sample_images(epoch_count, model, noise_schedule, device)



if __name__ == '__main__':
    print(f"Inizio training per {EPOCHS} epoche...")
    
    try:
        for _ in range(EPOCHS):
            training_epoch(dataloader)
            
    except KeyboardInterrupt:
        print("\nTraining interrotto manualmente. Salvataggio ultimo stato...")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_interrupted.pth"))
        print("Salvataggio di emergenza completato.")