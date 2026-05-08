import torch
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image, make_grid

# Setup path e import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.Network import ConditionalNetwork
from model.NoiseSchedule import NoiseSchedule
try: from config import *
except ImportError: pass

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./weights/diffusion/checkpoint_ep100.pth"
L = 1000
OUTPUT_NAME = "test_grid_8x8_ddim_fast.png"
GUIDANCE_SCALE = 1.5 # Leggermente aumentata per evidenziare gli attributi
SAMPLES_PER_COMBO = 8 
NUM_INFERENCE_STEPS = 50 # Numero di passi ridotto per Fast DDIM

def load_model_safe(model, path, device):
    if not os.path.exists(path):
        print(f"ERRORE: File non trovato {path}")
        return False
        
    print(f"Caricamento {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if 'ema_shadow' in checkpoint:
        print(">>> Trovati pesi EMA! Carico configurazione ottimizzata.")
        model.load_state_dict(checkpoint['ema_shadow'], strict=False)
    elif 'model_state_dict' in checkpoint:
        print(">>> Carico pesi standard.")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    return True

def generate_test_grid():
    # 1. Inizializza Modello
    model = ConditionalNetwork(L=L).to(DEVICE)
    if not load_model_safe(model, CHECKPOINT_PATH, DEVICE):
        return
    model.eval()
    noise_schedule = NoiseSchedule(L, device=DEVICE)
    
    # 2. Definiamo le 8 Combinazioni
    attr_combinations = [
        ([0, 0, 0], "Donna Vecchia Seria"),
        ([0, 0, 1], "Donna Giovane Seria"),
        ([0, 1, 0], "Donna Vecchia Felice"),
        ([0, 1, 1], "Donna Giovane Felice"),
        ([1, 0, 0], "Uomo Vecchio Serio"),
        ([1, 0, 1], "Uomo Giovane Serio"),
        ([1, 1, 0], "Uomo Vecchio Felice"),
        ([1, 1, 1], "Uomo Giovane Felice")
    ]
    
    num_combinations = len(attr_combinations)
    total_samples = num_combinations * SAMPLES_PER_COMBO 
    
    print(f"Preparazione batch da {total_samples} immagini...")

    attrs_base = torch.zeros(num_combinations, 3)
    row_labels = []

    for i, (vec, text) in enumerate(attr_combinations):
        attrs_base[i] = torch.tensor(vec).float()
        row_labels.append(text)
    
    attrs = attrs_base.repeat_interleave(SAMPLES_PER_COMBO, dim=0).to(DEVICE)

    # 3. Sampling Fast DDIM
    print(f"Generazione in corso (Steps: {NUM_INFERENCE_STEPS}, Guidance: {GUIDANCE_SCALE})...")
    
    with torch.no_grad():
        x = torch.randn(total_samples, 3, 64, 64).to(DEVICE) 
        
        # --- CALCOLO TIMESTEPS PER FAST SAMPLING ---
        # Creiamo una lista spaziata, es: [980, 960, ..., 20, 0]
        step_ratio = L // NUM_INFERENCE_STEPS
        timesteps = list(range(0, L, step_ratio))
        timesteps = list(reversed(timesteps))
        
        # --- LOOP FAST DDIM ---
        for i, t_index in enumerate(timesteps):
            # t_index è il tempo reale (es. 980).
            t = torch.full((total_samples,), t_index, device=DEVICE, dtype=torch.long)
            
            # A. Predizione con Guidance
            if GUIDANCE_SCALE > 1.0:
                eps_cond = model(x, t, attrs, drop_condition=False)
                eps_uncond = model(x, t, attrs, drop_condition=True)
                eps = eps_uncond + GUIDANCE_SCALE * (eps_cond - eps_uncond)
            else:
                eps = model(x, t, attrs, drop_condition=False)

            # B. Recupero Alpha Corrente e Precedente (Next Step)
            alpha_bar_t = noise_schedule.alpha[t_index]
            
            # Guardiamo al prossimo elemento della lista ridotta per trovare il prev_alpha
            if i < len(timesteps) - 1:
                prev_t_index = timesteps[i + 1]
                alpha_bar_prev = noise_schedule.alpha[prev_t_index]
            else:
                # Ultimo step (t=0), arriviamo a immagine pulita
                alpha_bar_prev = torch.tensor(1.0).to(DEVICE)
            
            # C. Equazione DDIM Update
            denom = torch.sqrt(alpha_bar_t) + 1e-8 # epsilon per evitare divisioni per zero
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / denom
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0) # Clipping fondamentale per stabilità
            
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if i % 10 == 0: 
                print(f"Step {i}/{len(timesteps)} (t={t_index})")

        # Post-processing
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        x = x.cpu()

    # 4. Visualizzazione Griglia 8x8 con Matplotlib
    print("Creazione griglia...")
    fig, axes = plt.subplots(num_combinations, SAMPLES_PER_COMBO, figsize=(20, 24), squeeze=False)
    
    for row in range(num_combinations):
        axes[row, 0].set_ylabel(row_labels[row], rotation=0, size='large', labelpad=80, fontweight='bold')
        
        for col in range(SAMPLES_PER_COMBO):
            idx = row * SAMPLES_PER_COMBO + col
            
            img = x[idx].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_xticks([]) 
            axes[row, col].set_yticks([])
            
            if row == 0:
                axes[row, col].set_title(f"Sample {col+1}")

    plt.tight_layout()
    save_dir = Path("output/diffusion") 
    save_path = save_dir / OUTPUT_NAME

    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=100)
    plt.close()
    
    print(f"Griglia salvata con successo: {OUTPUT_NAME}")

if __name__ == "__main__":
    generate_test_grid()