import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from torchvision.utils import make_grid

# --- PARAMETRI ---
EXP_NAME = "cvae_final_VGG128sqrt_balanced_sl4"
LATENT_DIM = 128 
ATTR_INDICES = [20, 31, 39] # Male, Smiling, Young
NUM_ATTR = len(ATTR_INDICES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.5  # T < 1.0 per maggiore nitidezza
SAMPLES_PER_ROW = 8 

# --- CONFIGURAZIONE ---
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

import config
from src.models.cvae import CelebACVAE

def generate_table():
    checkpoint_path = config.MODEL_PATH / EXP_NAME / "model.pth"
    output_dir = config.OUTPUT_DIR_PATH / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        print(f"Errore: Modello non trovato in {checkpoint_path}")
        return

    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
    model.eval()

    scenari = [
        {"desc": "DONNA - SORRIDENTE - GIOVANE", "attr": [0., 1., 1.]},
        {"desc": "DONNA - SORRIDENTE - ANZIANA", "attr": [0., 1., 0.]},
        {"desc": "DONNA - NON SORRIDENTE - GIOVANE",      "attr": [0., 0., 1.]},
        {"desc": "DONNA - NON SORRIDENTE - ANZIANA",     "attr": [0., 0., 0.]},
        {"desc": "UOMO - SORRIDENTE - GIOVANE",  "attr": [1., 1., 1.]},
        {"desc": "UOMO - SORRIDENTE - ANZIANO",  "attr": [1., 1., 0.]},
        {"desc": "UOMO - NON SORRIDENTE - GIOVANE",       "attr": [1., 0., 1.]},
        {"desc": "UOMO - NON SORRIDENTE - ANZIANO",      "attr": [1., 0., 0.]},
    ]

    fig, axes = plt.subplots(len(scenari), 1, figsize=(18, 22))
    
    print(f"Generazione tabella ordinata per {EXP_NAME} con T={TEMPERATURE}...")

    with torch.no_grad():
        for i, scenario in enumerate(scenari):
            # Preparazione condizionamento
            cond = torch.tensor([scenario["attr"]]).to(DEVICE).float()
            cond_expanded = cond.expand(SAMPLES_PER_ROW, -1)
            
            # Generazione vettore latente con Temperatura
            z = torch.randn(SAMPLES_PER_ROW, LATENT_DIM).to(DEVICE) * TEMPERATURE
            
            # --- DECODER ---
            z_cond = torch.cat((z, cond_expanded), dim=1)
            y = model.dec_input(z_cond).view(-1, 512, 4, 4)
            
            y = model.up1(model._inject_cond(y, cond_expanded))
            y = model.up2(model._inject_cond(y, cond_expanded))
            y = model.up3(model._inject_cond(y, cond_expanded))
            y = model.up4(model._inject_cond(y, cond_expanded))
            
            grid = make_grid(y.cpu(), nrow=SAMPLES_PER_ROW, padding=4, normalize=True)
            img = grid.permute(1, 2, 0).numpy()
            
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(scenario["desc"], fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout(pad=3.0)
    save_path = output_dir / f"grid_{EXP_NAME}_T{TEMPERATURE}.png"
    plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"Tabella salvata con successo in: {save_path}")

if __name__ == "__main__":
    generate_table()