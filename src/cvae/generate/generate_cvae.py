import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from torchvision.utils import make_grid

# --- PARAMETRI ---
EXP_NAME = "cvae"
LATENT_DIM = 128 
ATTR_INDICES = [20, 31, 39] # Male, Smiling, Young
NUM_ATTR = len(ATTR_INDICES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 1.0  # Parametro di "Campionamento": controlla la varietà dei volti generati
SAMPLES_PER_ROW = 8 

# --- CONFIGURAZIONE ---
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT))

import config
from src.cvae.model.cvae import CelebACVAE

def generate_table():
    checkpoint_path = config.MODEL_PATH / EXP_NAME / "model.pth"
    output_dir = config.OUTPUT_DIR_PATH / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        print(f"Errore: Modello non trovato in {checkpoint_path}")
        return

    # Inizializza l'architettura CVAE
    model = CelebACVAE(latent_dim=LATENT_DIM, num_attr=NUM_ATTR).to(DEVICE)

    # Carica i pesi addestrati. map_location assicura che il modello funzioni anche se 
    # addestrato su GPU e caricato su CPU (o viceversa).
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
    
    # Mette il modello in modalità EVAL: disabilita il Dropout e fissa le statistiche dei BatchNorm
    model.eval()

    scenari = [
        {"desc": "DONNA - SORRIDENTE - GIOVANE", "attr": [0., 1., 1.]},
        {"desc": "DONNA - SORRIDENTE - ANZIANA", "attr": [0., 1., 0.]},
        {"desc": "DONNA - NON SORRIDENTE - GIOVANE", "attr": [0., 0., 1.]},
        {"desc": "DONNA - NON SORRIDENTE - ANZIANA", "attr": [0., 0., 0.]},
        {"desc": "UOMO - SORRIDENTE - GIOVANE", "attr": [1., 1., 1.]},
        {"desc": "UOMO - SORRIDENTE - ANZIANO", "attr": [1., 1., 0.]},
        {"desc": "UOMO - NON SORRIDENTE - GIOVANE", "attr": [1., 0., 1.]},
        {"desc": "UOMO - NON SORRIDENTE - ANZIANO", "attr": [1., 0., 0.]},
    ]

    # Crea una figura con una riga per ogni scenario
    fig, axes = plt.subplots(len(scenari), 1, figsize=(18, 22))
    
    print(f"Generazione tabella ordinata per {EXP_NAME} con T={TEMPERATURE}...")

    with torch.no_grad():
        for i, scenario in enumerate(scenari):
            # Converte la lista di attributi in un tensore e lo espande per coprire l'intera riga (Batch)
            cond = torch.tensor([scenario["attr"]]).to(DEVICE).float()
            cond_expanded = cond.expand(SAMPLES_PER_ROW, -1)
            
            # --- GENERAZIONE RUMORE ---
            # Campioniamo rumore gaussiano e lo scaliamo per la Temperatura.
            # Questo z rappresenta l'identità del volto non catturata dagli attributi.
            z = torch.randn(SAMPLES_PER_ROW, LATENT_DIM).to(DEVICE) * TEMPERATURE
            
            # --- DECODER ---
             # Uniamo il rumore z con le condizioni scelte (concatenazione lungo i canali)
            z_cond = torch.cat((z, cond_expanded), dim=1)
            y = model.dec_input(z_cond).view(-1, 512, 4, 4)
            
            # Ricostruzione progressiva: iniettiamo gli attributi ad ogni upsampling.
            # Questo garantisce che il condizionamento sia mantenuto forte fino all'output finale.
            y = model.up1(model._inject_cond(y, cond_expanded))
            y = model.up2(model._inject_cond(y, cond_expanded))
            y = model.up3(model._inject_cond(y, cond_expanded))
            y = model.up4(model._inject_cond(y, cond_expanded))
            
            # --- VISUALIZZAZIONE ---
            # make_grid crea un'unica immagine affiancando i volti generati nella riga
            grid = make_grid(y.cpu(), nrow=SAMPLES_PER_ROW, padding=4, normalize=True)
            # Riordina i canali da [C, H, W] (PyTorch) a [H, W, C] (Numpy/Matplotlib)
            img = grid.permute(1, 2, 0).numpy()
            
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(scenario["desc"], fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout(pad=3.0) # Evita sovrapposizioni tra le righe
    save_path = output_dir / f"cvae_T{TEMPERATURE}.png"
    plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"Tabella salvata con successo in: {save_path}")

if __name__ == "__main__":
    generate_table()