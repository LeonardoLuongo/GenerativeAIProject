import sys
import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image, make_grid

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_path)

from vqvae.model.vq_vae import VQVAE
from mamba.model.mamba_vqvae import MambaVQVAE

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VQ_CHECKPOINT = "checkpoints/vqvae_16x16_2_epoch_50.pth"
MAMBA_CHECKPOINT = "checkpoints/mamba_vqvae_snake_cfg_k17_epoch_100.pth" # Usa il tuo ultimo checkpoint

# Configurazione identica al training
MAMBA_CONFIG = {
    'vocab_size': 1024,
    'dim': 512,
    'state_size': 16,
    'num_layers': 16,
    'expansion': 2.0,
    'kernel_size': 17,
    'dropout_prob': 0.0
}

def load_models():
    print("--- Caricamento Modelli ---")
    vqvae = VQVAE(num_embeddings=1024, embedding_dim=256).to(DEVICE)
    vqvae.load_state_dict(torch.load(VQ_CHECKPOINT, map_location=DEVICE)['model_state_dict'])
    vqvae.eval()
    
    mamba = MambaVQVAE(**MAMBA_CONFIG).to(DEVICE)
    mamba.load_state_dict(torch.load(MAMBA_CHECKPOINT, map_location=DEVICE))
    mamba.eval()
    return vqvae, mamba

def main():
    vqvae, mamba = load_models()
    
    # 1. Configuriamo le 8 categorie
    # Ordine: [Maschio, Sorriso, Giovane]
    categories = [
        [0, 0, 0], # Donna, Seria, Anziana
        [0, 0, 1], # Donna, Seria, Giovane
        [0, 1, 0], # Donna, Sorride, Anziana
        [0, 1, 1], # Donna, Sorride, Giovane
        [1, 0, 0], # Uomo, Serio, Anziano
        [1, 0, 1], # Uomo, Serio, Giovane
        [1, 1, 0], # Uomo, Sorride, Anziano
        [1, 1, 1]  # Uomo, Sorride, Giovane
    ]
    
    labels_text = [
        "F NoSmile Old", "F NoSmile Young", "F Smile Old", "F Smile Young",
        "M NoSmile Old", "M NoSmile Young", "M Smile Old", "M Smile Young"
    ]

    NUM_VARIANTS = 16 # Quante immagini per ogni categoria
    
    # Creiamo il batch gigante: (8 categorie * 8 varianti) = 64 immagini
    attr_list = []
    for cat in categories:
        for _ in range(NUM_VARIANTS):
            attr_list.append(cat)
            
    attr_tensor = torch.tensor(attr_list, dtype=torch.long).to(DEVICE) # Shape (64, 3)
    
    print(f"Generazione di {len(attr_tensor)} immagini in parallelo...")
    print("Attendere, la generazione richiede tempo (steps=256)...")
    
    with torch.no_grad():
        # --- GENERAZIONE MAMBA ---
        # CFG Scale alto (4.0 o 5.0) migliora la nitidezza e l'aderenza agli attributi
        # Temperature < 1.0 riduce gli artefatti
        indices_snake = mamba.generate( 
            attr_tensor, 
            steps=256, 
            temperature=1.0, 
            cfg_scale=1.5 
        )
        
        # --- RETTIFICA INDICI (Snake -> Raster) ---
        indices_raster = MambaVQVAE.snake_process(indices_snake, side_len=16)
        
        # --- DECODIFICA VQ-VAE ---
        images = vqvae.decode_from_indices(indices_raster) # (64, 3, 64, 64)

    # Denormalizza: [-1, 1] -> [0, 1]
    images = images.cpu() * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)
    
    # --- 1. SALVATAGGIO SU DISCO (Alta Qualità) ---
    # make_grid unisce tutto in un'unica immagine
    # nrow=8 significa che avremo 8 colonne (quindi ogni riga è una categoria)
    grid_img = make_grid(images, nrow=NUM_VARIANTS, padding=2, pad_value=1) # pad_value=1 è bianco
    
    output_filename = "mamba_mosaic_results.png"
    save_image(grid_img, output_filename)
    print(f"\n✅ Immagine salvata come: {output_filename}")
    print("   Aprila con un visualizzatore immagini per vederla alla risoluzione reale!")

    # --- 2. VISUALIZZAZIONE A SCHERMO ---
    plt.figure(figsize=(12, 12)) # Finestra grande
    plt.imshow(grid_img.permute(1, 2, 0).numpy(), interpolation='nearest')
    
    # Aggiungiamo le etichette a sinistra
    # Calcolo posizione approssimativa delle righe
    h = grid_img.shape[1]
    step = h / 8
    for i, label in enumerate(labels_text):
        plt.text(-50, (i * step) + (step/2), label, fontsize=10, verticalalignment='center', horizontalalignment='left')

    plt.title(f"Mamba Generation: 8 Categorie x {NUM_VARIANTS} Varianti", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()