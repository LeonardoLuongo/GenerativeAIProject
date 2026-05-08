import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import os
from Network import ConditionalNetwork
from NoiseSchedule import NoiseSchedule
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from config import *
except ImportError:
    print("Attenzione: config.py non trovato, uso parametri locali.")

if torch.cuda.is_available():
    # Prova a forzare l'uso della prima GPU visibile assegnata da SLURM
    device = torch.device("cuda:0")
    print(f"CUDA Disponibile. Device count: {torch.cuda.device_count()}")
    print(f"Device corrente: {torch.cuda.current_device()}")
    print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA NON disponibile. Uso CPU.")

# Parametri del Modello e Training
L = 1000                # Step di diffusione
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01   # Regolarizzazione
WARMUP_EPOCHS = 5
EPOCHS = 100

# Percorsi
CHECKPOINT_DIR = './checkpoints'
DATA_ROOT = '/home/pfoggia/GenerativeAI/CELEBA'
ATTR_INDICES = [20, 31, 39] 

transform = transforms.Compose([
    transforms.CenterCrop(178),             # Ritaglia il quadrato centrale
    transforms.Resize((64, 64)),            # Ridimensiona a 64x64
    transforms.ToTensor(),                  # Converte in tensore [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),   # Normalizza in [-1, 1]
                         (0.5, 0.5, 0.5))
])
print(f"Caricamento dataset CelebA da: {DATA_ROOT}")
dataset = CelebA(root=DATA_ROOT, split='train', target_type='attr', transform=transform, download=False)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
noise_schedule = NoiseSchedule(L, device=device)

class EMA:
    '''
    Invece di usare direttamente i pesi dell'ultimo step di training (che possono oscillare molto),
    questa classe mantiene una copia "ombra" (shadow) dei pesi che rappresenta una media mobile
    esponenziale nel tempo.
    Filtra il rumore stocastico introdotto dai mini-batch e dall'ottimizzatore.
    '''
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        # Aggiorna i pesi ombra
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        # Salva i pesi attuali nel backup e carica i pesi EMA nel modello.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        # Ripristina i pesi originali dal backup.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# Creiamo la cartella per salvare i modelli se non esiste
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

print(f"Inizializzazione su device: {device}")



# Creazione Modello
print("Creazione modello...")
# Passiamo L al costruttore come abbiamo modificato prima
model = ConditionalNetwork(L=L, cfg_dropout=0.1)
model = model.to(device)

ema = EMA(model, decay=0.999)

# Optimizer AdamW migliore per batch grandi e Loss
optimizer = optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

# Variabile globale per contare le epoche
epoch_count = 0

def get_lr_multiplier(epoch):
    """
    Gestisce l'evoluzione del Learning Rate.
    Strategia: Linear Warmup -> Cosine Decay.
    """
    # WARMUP: Primi step (es. 0-5 epoche)
    # Serve a stabilizzare il training quando i pesi sono ancora random.
    # Il LR sale linearmente da 0 al valore massimo.
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        # Il LR scende seguendo una curva coseno morbida fino a zero.
        # All'inizio serve alto per imparare i concetti generali.
        # Alla fine serve basso per rifinire i dettagli fini senza "saltare" via dal minimo.
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

def save_sample_images(epoch, model, noise_schedule, device):
    """
    Generazione immagini usando DDIM (Deterministico).
    Produce risultati più nitidi e coerenti rispetto a DDPM.
    """
    # Carichiamo i pesi "mediati" (EMA) che sono più stabili.
    ema.apply_shadow()
    model.eval()
    with torch.no_grad():
        n_samples = 8

        # Impostiamo quanti passi vogliamo fare (es. 50 invece di 1000)
        num_inference_steps = 50

        # 1. Rumore Iniziale (Latent Code)
        # In DDIM, se fissiamo questo x iniziale, otterremo sempre la stessa faccia identica.
        x = torch.randn(n_samples, 3, 64, 64).to(device)
        
        # Attributi di test
        attrs = torch.zeros(n_samples, 3).to(device)
        attrs[:4] = torch.tensor([1, 1, 1]) # Uomo, Sorriso, Giovane
        attrs[4:] = torch.tensor([0, 0, 0]) # Donna, No Sorriso, non Giovane
        
        # Scala per Classifier-Free Guidance (CFG)
        guidance_scale = 1.3

        # 1. Calcoliamo la lista dei timestep da visitare (es. [980, 960, ..., 0])
        step_ratio = L // num_inference_steps
        timesteps = list(range(0, L, step_ratio))
        timesteps = list(reversed(timesteps)) # Dobbiamo andare dal rumore (980) verso la pulizia (0)
        
        # 2. Ciclo di Sampling DDIM
        for i, t_index in enumerate(timesteps):
            # t_index è il tempo reale (es. 980). Creiamo il tensore per il batch.
            t = torch.full((n_samples,), t_index, device=device, dtype=torch.long)
            
            # A. Predizione del rumore condizionata
            eps_cond = model(x, t, attrs, drop_condition=False)
            # Predizione non condizionata
            eps_uncond = model(x, t, attrs, drop_condition=True)
            # Formula per CFG
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # Alpha corrente (al tempo t_index, es. 980)
            # Nota: noise_schedule.alpha contiene il prodotto cumulativo (alpha_bar)
            alpha_bar_t = noise_schedule.alpha[t_index]
            
            # Alpha del PROSSIMO passo
            if i < len(timesteps) - 1:
                # Se non siamo alla fine, guardiamo il prossimo elemento nella lista timesteps
                prev_t_index = timesteps[i + 1]
                alpha_bar_prev = noise_schedule.alpha[prev_t_index]
            else:
                # Siamo all'ultimo passo (t=0), il "precedente" è l'immagine pura (alpha=1.0)
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
    total_loss = 0.0
        
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_count + 1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f"{'='*60}")
    
    for step, (x, all_attributes) in enumerate(dataloader):
        # Spostiamo i dati sulla GPU
        x = x.to(device)
        attributes = all_attributes[:, ATTR_INDICES].float().to(device)
        
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
        
        # LOSS PESATA PER TIMESTEP (Min-SNR weighting)
        # Timestep bassi (immagine pulita) = peso più alto
        # Timestep alti (rumore puro) = peso più basso
        snr = noise_schedule.alpha[t] / (1 - noise_schedule.alpha[t] + 1e-8)
        snr_clamped = torch.clamp(snr, min=0.5, max=2.0)
        
        # Loss base
        loss_simple = F.mse_loss(g, eps, reduction='none')
        loss_simple = loss_simple.mean(dim=[1, 2, 3])  # (B,)
        
        # Applica peso
        loss = (loss_simple * snr_clamped).mean()

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

        total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"Step [{step:4d}/{len(dataloader)}] | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    epoch_count += 1
    
    print(f"\n>>> Epoch {epoch_count} completata | Avg Loss: {avg_loss:.4f}")
    
    # SALVATAGGIO CHECKPOINT
    checkpoint = {
        'epoch': epoch_count,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{epoch_count}.pth"))
    
    # Sample ogni 5 epoche
    if epoch_count % 5 == 0 or epoch_count == 1:
        save_sample_images(epoch_count, model, noise_schedule, device)
    
    scheduler.step()



if __name__ == '__main__':
    print(f"Inizio training per {EPOCHS} epoche...")
    
    # --- LOGICA DI RIPRESA (RESUME) ---
    # Imposta qui il nome del file da cui vuoi ripartire (es. epoch 65)
    resume_path = os.path.join(CHECKPOINT_DIR, "checkpoint_ep67.pth")
    
    start_epoch = 0 # Default se non trova nulla
    
    if os.path.exists(resume_path):
        print(f"\n>>> Trovato checkpoint: {resume_path}")
        print(">>> Caricamento stato del training...")
        
        # Carichiamo il dizionario completo
        checkpoint = torch.load(resume_path, map_location=device)
        
        # 1. Ripristina i pesi del Modello
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. Ripristina l'EMA (Fondamentale per la qualità finale)
        if 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
            print(" -> EMA ripristinato.")
            
        # 3. Ripristina Optimizer e Scheduler 
        # (Questo evita che il Learning Rate torni alto all'inizio)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 4. Imposta l'epoca di partenza
        start_epoch = checkpoint['epoch']
        epoch_count = start_epoch # Aggiorniamo la variabile globale
        
        print(f">>> Training ripristinato! Si riparte dall'epoca {start_epoch + 1}.\n")
    else:
        print(">>> Nessun checkpoint trovato o file non esistente. Si parte da ZERO.\n")

    try:
        # Modifichiamo il loop per usare range(start, end)
        for _ in range(start_epoch, EPOCHS):
            training_epoch(dataloader)
            
    except KeyboardInterrupt:
        print("\nTraining interrotto manualmente. Salvataggio di emergenza...")
        # Salviamo un checkpoint completo anche in caso di interruzione
        checkpoint = {
            'epoch': epoch_count,
            'model_state_dict': model.state_dict(),
            'ema_shadow': ema.shadow,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': 0.0, # Valore dummy
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "model_interrupted.pth"))
        print("Salvataggio di emergenza completato.")