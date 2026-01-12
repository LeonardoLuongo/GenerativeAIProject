import sys
import os
# Aggiungi src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Importa configurazioni e modello
# Assicurati che model/mamba/mamba_block.py sia quello "LENTO" ora
from model.mamba.mamba_image_model import MambaImageModel

# --- CONFIGURAZIONE ---
# Devono corrispondere a quelli usati nel training
EMBEDDING_DIMENSION = 512 
STATE_SIZE = 16
EXPANSION = 1.1
CONV_KERNEL = 4
MAMBA_LAYERS = 5
CHECKPOINT_PATH = "checkpoints/mamba_epoch_1.pth" # CAMBIA QUESTO con il tuo miglior checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    print(f"Caricamento modello da {path} su {DEVICE}...")
    model = MambaImageModel(
        dim=EMBEDDING_DIMENSION,
        state_size=STATE_SIZE,
        expansion=EXPANSION,
        conv_kernel=CONV_KERNEL,
        num_layers=MAMBA_LAYERS,
        num_attr=3
    ).to(DEVICE)
    
    # Carica i pesi
    checkpoint = torch.load(path, map_location=DEVICE)
    # Gestione caso salvataggio intero vs state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("Modello caricato con successo!")
    return model

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Errore: Checkpoint non trovato in {CHECKPOINT_PATH}")
        return

    model = load_model(CHECKPOINT_PATH)
    model.eval()

    print("\n--- GENERATORE VOLTI CELEBA (MAMBA) ---")
    print("Inserisci gli attributi per generare un volto.")
    print("Usa 1 per SI, 0 per NO.")
    
    while True:
        try:
            print("\n-------------------------------------------")
            male_str = input("1. È maschio? (1/0): ")
            smiling_str = input("2. Sorride? (1/0): ")
            young_str = input("3. È giovane? (1/0): ")
            
            male = int(male_str)
            smiling = int(smiling_str)
            young = int(young_str)
            
            # Crea il tensore attributi (Batch size 1)
            # Ordine: [Male, Smiling, Young] come nel training
            attrs = torch.tensor([[male, smiling, young]], dtype=torch.long).to(DEVICE)
            
            print(f"\nInizio generazione... (Attendere, processo autoregressivo pixel-by-pixel)")
            
            # === GENERAZIONE CON PROGRESS BAR ===
            # Copiamo la logica di generate() ma aggiungiamo tqdm qui per visualizzazione
            with torch.no_grad():
                # Questa funzione è lenta (4096 step sequenziali)
                img_indices = model.generate(attrs) # (1, 4096)
            
            # 3. Ricostruzione immagine
            img_grid = img_indices.view(64, 64).cpu().numpy()
            
            # 4. Visualizzazione
            plt.figure(figsize=(5, 5))
            plt.imshow(img_grid, cmap='gray', vmin=0, vmax=255)
            title = f"Maschio:{male} Sorriso:{smiling} Giovane:{young}"
            plt.title(title)
            plt.axis('off')
            plt.show()
            
            cont = input("Generare un altro? (s/n): ")
            if cont.lower() != 's':
                break
                
        except ValueError:
            print("Input non valido. Inserisci solo 0 o 1.")
        except KeyboardInterrupt:
            print("\nUscita.")
            break

if __name__ == "__main__":
    main()