import sys
import os
import torch
import matplotlib.pyplot as plt
import torchvision

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.vq_vae.vq_vae import VQVAE
from model.mamba.mamba_vqvae import MambaVQVAE

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VQ_PATH = "checkpoints/vqvae_16x16_2_epoch_50.pth"     # Il tuo VQ-VAE
MAMBA_PATH = "checkpoints/mamba_vq_epoch_20.pth"       # Il tuo Mamba addestrato

def load_models():
    # 1. VQ-VAE
    vq = VQVAE(num_embeddings=1024, embedding_dim=256).to(DEVICE)
    vq.load_state_dict(torch.load(VQ_PATH, map_location=DEVICE)['model_state_dict'])
    vq.eval()
    
    # 2. Mamba
    mamba = MambaVQVAE(vocab_size=1024, dim=512, state_size=16, num_layers=6).to(DEVICE)
    mamba.load_state_dict(torch.load(MAMBA_PATH, map_location=DEVICE))
    mamba.eval()
    
    return vq, mamba

def main():
    vq_model, mamba_model = load_models()
    
    print("\n--- GENERATORE MAMBA + VQ-VAE ---")
    while True:
        try:
            print("\nInserisci attributi (1=Si, 0=No):")
            male = int(input("Maschio? "))
            smile = int(input("Sorriso? "))
            young = int(input("Giovane? "))
            
            # Batch size 1
            attrs = torch.tensor([[male, smile, young]]).to(DEVICE)
            
            print("Generazione in corso (256 step)...")
            with torch.no_grad():
                # 1. Mamba genera gli indici (token)
                generated_indices = mamba_model.generate(attrs, steps=256) # (1, 256)
                
                # 2. VQ-VAE decodifica gli indici in immagine
                # decode_from_indices si aspetta (B, H, W) o (B, SeqLen) gestito internamente
                generated_img = vq_model.decode_from_indices(generated_indices)
                
            # Visualizzazione
            img_tensor = generated_img.squeeze(0).cpu() # (3, 64, 64)
            img_tensor = img_tensor * 0.5 + 0.5         # Denormalize
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            plt.figure(figsize=(4,4))
            plt.imshow(img_tensor.permute(1, 2, 0))
            plt.title(f"M:{male} S:{smile} Y:{young}")
            plt.axis('off')
            plt.show()
            
        except ValueError:
            break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()