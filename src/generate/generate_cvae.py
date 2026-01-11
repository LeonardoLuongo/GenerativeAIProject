import torch
from torchvision.utils import save_image
import sys
from pathlib import Path

# --- PARAMETRI (Devono essere uguali al train) ---
EXP_NAME = "cvae_v2_mse_latent256_beta05"
LATENT_DIM = 512 
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
    model.load_state_dict(torch.load(str(checkpoint_path)))
    model.eval()

    # 3. Definiamo le combinazioni di test [Uomo, Sorriso, Giovane]
    # 1.0 = Sì, 0.0 = No
    scenari = {
        "donna_sorridente_giovane": [0., 1., 1.],
        "uomo_non_sorridente_giovane": [1., 0., 1.],
        "donna_non_sorridente_anziana": [0., 0., 0.],
        "uomo_sorridente_anziano": [1., 1., 0.]
    }

    print("Generazione immagini in corso...")

    with torch.no_grad():
        for nome, attributi in scenari.items():
            # Creiamo un vettore di attributi e lo spostiamo su GPU
            cond = torch.tensor([attributi]).to(DEVICE)
            
            # Generiamo 16 variazioni diverse (z casuali) per lo stesso identico attributo
            z = torch.randn(16, LATENT_DIM).to(DEVICE)
            
            # Espandiamo il condizionamento per tutti i 16 campioni
            cond_expanded = cond.expand(16, -1)
            
            # Passiamo al decoder: concateniamo z e attributi
            z_cond = torch.cat((z, cond_expanded), dim=1)
            
            # Generazione pixel
            samples = model.decoder_input(z_cond)
            samples = model.decoder_conv(samples).cpu()
            
            save_path = output_dir / f"risultato_{nome}.png"
            save_image(samples, save_path, nrow=4, normalize=True)
            print(f"Salvato: {save_path}")

if __name__ == "__main__":
    generate()