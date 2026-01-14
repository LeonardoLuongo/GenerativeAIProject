import torch
import torchvision.utils as vutils
import os
from Network import ConditionalNetwork
from NoiseSchedule import NoiseSchedule
from config import *

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = BETTER_MODEL
L = 1000
OUTPUT_NAME = "test_grid_ep43.png"

def generate_attribute_test():
    print(f"--- Inizio Test Attributi su Device: {DEVICE} ---")
    
    # 1. Carichiamo il modello
    print("Caricamento modello...")
    model = ConditionalNetwork(L=L).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Checkpoint caricato: {CHECKPOINT_PATH}")
    else:
        print(f"ERRORE: Non trovo il file {CHECKPOINT_PATH}")
        return

    model.eval()
    noise_schedule = NoiseSchedule(L, device=DEVICE)

    # 2. Definiamo le 8 Combinazioni (Tabella della Verità)
    # Ordine vettore: [Uomo, Sorriso, Giovane]
    # 1 = Sì, 0 = No
    attributes_list = [
        [1, 1, 1], # 1. Uomo, Sorriso, Giovane
        [1, 1, 0], # 2. Uomo, Sorriso, Vecchio
        [1, 0, 1], # 3. Uomo, No Sorriso, Giovane
        [1, 0, 0], # 4. Uomo, No Sorriso, Vecchio
        [0, 1, 1], # 5. Donna, Sorriso, Giovane
        [0, 1, 0], # 6. Donna, Sorriso, Vecchio
        [0, 0, 1], # 7. Donna, No Sorriso, Giovane
        [0, 0, 0]  # 8. Donna, No Sorriso, Vecchio
    ]
    
    descriptions = [
        "Uomo Giovane Felice", "Uomo Vecchio Felice", 
        "Uomo Giovane Serio", "Uomo Vecchio Serio",
        "Donna Giovane Felice", "Donna Vecchia Felice", 
        "Donna Giovane Seria", "Donna Vecchia Seria"
    ]

    n_samples = 8
    attrs_tensor = torch.tensor(attributes_list, dtype=torch.float32).to(DEVICE)

    # 3. Generazione (DDIM Loop)
    with torch.no_grad():
        # Usiamo lo stesso rumore di partenza per vedere come gli attributi modificano la STESSA faccia base
        # Se vuoi facce diverse, togli il commento alla riga sotto e commenta quella del 'fixed_noise'
        x = torch.randn(n_samples, 3, 64, 64).to(DEVICE) 
        
        # Opzione: Stessa "anima" (noise), attributi diversi. 
        # Questo è il test più potente per vedere se il modello controlla bene le feature.
        #fixed_noise = torch.randn(1, 3, 64, 64).to(DEVICE)
        #x = fixed_noise.repeat(n_samples, 1, 1, 1)

        print("Inizio sampling DDIM (pazientare)...")
        
        for i in range(L - 1, -1, -1):
            t = torch.tensor([i] * n_samples, device=DEVICE)
            
            # Predizione rumore
            eps = model(x, t, attrs_tensor)
            
            # Parametri DDIM
            alpha_bar_t = noise_schedule.alpha[i]
            alpha_bar_prev = noise_schedule.alpha[i - 1] if i > 0 else torch.tensor(1.0).to(DEVICE)

            # Calcolo x0 predetto
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Direzione verso xt-1
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            
            # Aggiornamento x
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if i % 100 == 0:
                print(f"Step {i}/{L}")

        # 4. Salvataggio
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        vutils.save_image(x, OUTPUT_NAME, nrow=4, padding=2, normalize=False)
        print(f"\nGenerazione completata! Immagine salvata come: {OUTPUT_NAME}")
        print("Griglia salvata (4 colonne):\n[Uomo G. Smile] [Uomo V. Smile] [Uomo G. Serio] [Uomo V. Serio]")
        print("[Donna G. Smile] [Donna V. Smile] [Donna G. Seria] [Donna V. Seria]")

if __name__ == "__main__":
    generate_attribute_test()