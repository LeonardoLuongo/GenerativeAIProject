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

# Aggiunta root al path
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

import config
from src.models.cvae import CelebACVAE

def generate():
    checkpoint_path = config.MODEL_PATH / EXP_NAME / "model.pth"
    output_dir = config.OUTPUT_DIR_PATH / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        print(f"Errore: Modello non trovato in {checkpoint_path}")
        return

    # Caricamento Modello
    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
    model.eval()

    # Combinazioni di test [Male, Smiling, Young]
    scenari = {
        "donna_sorridente_giovane": [0., 1., 1.],
        "uomo_non_sorridente_giovane": [1., 0., 1.],
        "donna_non_sorridente_anziana": [0., 0., 0.],
        "uomo_sorridente_anziano": [1., 1., 0.]
    }

    print(f"Generazione immagini in corso su {DEVICE}...")

    with torch.no_grad():
        for nome, attributi in scenari.items():
            # Creiamo un vettore di attributi e lo spostiamo su GPU
            cond = torch.tensor([attributi]).to(DEVICE)
            
            # Generiamo 16 variazioni diverse (z casuali)
            z = torch.randn(16, LATENT_DIM).to(DEVICE) 
            
            # Espandiamo il condizionamento per i 16 campioni
            cond_expanded = cond.expand(16, -1)
            
            # Concateniamo z e attributi
            z_cond = torch.cat((z, cond_expanded), dim=1)
            
            # --- LOGICA CORRETTA PER IL NUOVO MODELLO ---
            # Passiamo prima per il decoder_input
            x = model.decoder_input(z_cond)
            # Poi per il blocco decoder (le convoluzioni)
            samples = model.decoder(x).cpu()
            # --------------------------------------------
            
            save_path = output_dir / f"risultato_{nome}.png"
            # Nota: normalize=False perché la Sigmoid finale dà già valori [0,1]
            save_image(samples, save_path, nrow=4, normalize=False)
            print(f"Salvato: {save_path}")

if __name__ == "__main__":
    generate()