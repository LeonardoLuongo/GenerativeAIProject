import torch
from torchvision.utils import save_image
import sys
from pathlib import Path

# --- PARAMETRI (Devono essere uguali al train) ---
EXP_NAME = "cvae_balanced_sum_loss"
LATENT_DIM = 128 
ATTR_INDICES = [20, 31, 39] 
NUM_ATTR = len(ATTR_INDICES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PARAMETRO TEMPERATURA ---
# T = 1.0: Campionamento standard (facce più "medie")
# T > 1.0: Più varietà e diversità (es. 1.2, 1.5)
# T < 1.0: Più precisione, ma facce quasi identiche (es. 0.7)
TEMPERATURE = 1.3

# Aggiunta root al path
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

import config
from src.models.cvae import CelebACVAE

def generate():
    checkpoint_path = config.MODEL_PATH / EXP_NAME / "model.pth"
    # Creiamo una sottocartella specifica per i test con temperatura
    output_dir = config.OUTPUT_DIR_PATH / EXP_NAME / f"temp_{TEMPERATURE}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        print(f"Errore: Modello non trovato in {checkpoint_path}")
        return

    # 1. Caricamento Modello
    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
    model.eval()

    # 2. Combinazioni di test [Male, Smiling, Young]
    scenari = {
        "donna_sorridente_giovane": [0., 1., 1.],
        "uomo_non_sorridente_giovane": [1., 0., 1.],
        "donna_non_sorridente_anziana": [0., 0., 0.],
        "uomo_sorridente_anziano": [1., 1., 0.]
    }

    print(f"Generazione con Temperatura = {TEMPERATURE} su {DEVICE}...")

    with torch.no_grad():
        for nome, attributi in scenari.items():
            # Creiamo un vettore di attributi e lo spostiamo su GPU
            cond = torch.tensor([attributi]).to(DEVICE)
            
            # --- APPLICAZIONE TEMPERATURE CONTROL ---
            # Moltiplichiamo il rumore casuale per la temperatura.
            # Questo allarga la campana di Gauss, forzando il decoder a 
            # generare tratti meno "medi" e più originali.
            z = torch.randn(16, LATENT_DIM).to(DEVICE) * TEMPERATURE
            
            # Espandiamo il condizionamento per i 16 campioni
            cond_expanded = cond.expand(16, -1)
            
            # Concateniamo z e attributi
            z_cond = torch.cat((z, cond_expanded), dim=1)
            
            # Logica Decoder
            x = model.decoder_input(z_cond)
            samples = model.decoder(x).cpu()
            
            save_path = output_dir / f"risultato_{nome}_T{TEMPERATURE}.png"
            # normalize=False perché la Sigmoid finale dà già valori [0,1]
            save_image(samples, save_path, nrow=4, normalize=False)
            print(f"Salvato: {save_path}")

    print(f"\nGenerazione completata! Controlla la cartella: {output_dir}")

if __name__ == "__main__":
    # Puoi provare diverse temperature per trovare il punto di equilibrio
    # tra qualità (T bassa) e varietà (T alta).
    generate()