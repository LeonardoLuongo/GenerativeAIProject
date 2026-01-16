import torch
import os
import sys
import matplotlib.pyplot as plt

# Setup path e import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Network import ConditionalNetwork
from NoiseSchedule import NoiseSchedule
try: from config import *
except ImportError: pass

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/checkpoint_ep100.pth" # Usa il tuo ultimo checkpoint
L = 1000
OUTPUT_NAME = "test_grid_8x8_diverse_1.png"
GUIDANCE_SCALE = 1 
SAMPLES_PER_COMBO = 8 # Quanti volti diversi per ogni attributo (Colonne)

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
    
    # 2. Definiamo le 8 Combinazioni (Le Righe della nostra griglia)
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
    total_samples = num_combinations * SAMPLES_PER_COMBO # 8 * 8 = 64 immagini totali
    
    print(f"Preparazione batch da {total_samples} immagini...")

    # Creiamo i tensori per la batch gigante
    # attrs_base sarà (8, 3)
    attrs_base = torch.zeros(num_combinations, 3)
    row_labels = []

    for i, (vec, text) in enumerate(attr_combinations):
        attrs_base[i] = torch.tensor(vec).float()
        row_labels.append(text)
    
    # Ora "estendiamo" questi attributi ripetendoli SAMPLES_PER_COMBO volte
    # Esempio: [A, B] diventa [A, A, A... B, B, B...]
    # attrs diventa (64, 3)
    attrs = attrs_base.repeat_interleave(SAMPLES_PER_COMBO, dim=0).to(DEVICE)

    # 3. Sampling
    print(f"Generazione in corso (Guidance: {GUIDANCE_SCALE})...")
    with torch.no_grad():
        # Generiamo 64 rumori DIVERSI
        x = torch.randn(total_samples, 3, 64, 64).to(DEVICE) 
        
        for i in range(L - 1, -1, -1):
            t = torch.tensor([i] * total_samples, device=DEVICE)
            
            # Attenzione: Qui stiamo facendo forward su 64 immagini insieme.
            # Se la GPU va in "Out of Memory", dobbiamo ridurre la batch size.
            # Ma con 64x64 pixel e A100/3090/4090 non dovrebbero esserci problemi.
            
            if GUIDANCE_SCALE > 1.0:
                eps_cond = model(x, t, attrs, drop_condition=False)
                eps_uncond = model(x, t, attrs, drop_condition=True)
                eps = eps_uncond + GUIDANCE_SCALE * (eps_cond - eps_uncond)
            else:
                eps = model(x, t, attrs, drop_condition=False)

            alpha_bar = noise_schedule.alpha[i]
            alpha_bar_prev = noise_schedule.alpha[i-1] if i > 0 else torch.tensor(1.0).to(DEVICE)
            
            denom = torch.sqrt(alpha_bar) + 1e-8
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / denom
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if i % 200 == 0: print(f"Step {i}")

        # Post-processing
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        x = x.cpu()

    # 4. Visualizzazione Griglia 8x8 con Matplotlib
    print("Creazione griglia...")
    # Figura grande verticale per ospitare le 8 righe
    fig, axes = plt.subplots(num_combinations, SAMPLES_PER_COMBO, figsize=(20, 24))
    
    # axes è una matrice (8, 8) di subplot
    for row in range(num_combinations):
        # Imposta l'etichetta della riga (solo sul primo elemento della riga)
        axes[row, 0].set_ylabel(row_labels[row], rotation=0, size='large', labelpad=80, fontweight='bold')
        
        for col in range(SAMPLES_PER_COMBO):
            # L'indice nell'array piatto x è: riga * larghezza + colonna
            idx = row * SAMPLES_PER_COMBO + col
            
            img = x[idx].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_xticks([]) # Rimuove tacche assi
            axes[row, col].set_yticks([])
            
            # Opzionale: Titolo colonna sulla prima riga
            if row == 0:
                axes[row, col].set_title(f"Sample {col+1}")

    plt.tight_layout()
    plt.savefig(OUTPUT_NAME, dpi=100)
    plt.close()
    
    print(f"Griglia salvata con successo: {OUTPUT_NAME}")

if __name__ == "__main__":
    generate_test_grid()