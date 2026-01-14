import sys
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.vq_vae.vq_vae import VQVAE

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/vqvae_16x16_2_epoch_50.pth" # IL TUO CHECKPOINT VQ

def main():
    print(f"Test VQ-VAE da: {CHECKPOINT_PATH}")
    
    # 1. Carica Modello
    model = VQVAE(num_embeddings=1024, embedding_dim=256).to(DEVICE)
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # Gestione flessibile del caricamento pesi
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Errore caricamento pesi: {e}")
        return

    model.eval()

    # 2. Prendi un'immagine reale
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CelebA(root='./dataset', split='test', target_type='attr', transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    real_imgs, _ = next(iter(loader))
    real_imgs = real_imgs.to(DEVICE)
    
    # 3. Ricostruisci
    with torch.no_grad():
        # Passaggio completo
        recon_imgs, _, _ = model(real_imgs)
        
        # Passaggio tramite Indici (Simulazione Mamba)
        indices = model.get_codebook_indices(real_imgs)
        print(f"Esempio indici estratti: {indices[0, :10]}") # Vediamo se sono tutti uguali
        recon_from_indices = model.decode_from_indices(indices)

    # 4. Visualizza
    # Denormalizza: [-1, 1] -> [0, 1]
    def show(tensor, title):
        tensor = tensor.cpu() * 0.5 + 0.5
        tensor = tensor.clamp(0, 1)
        grid = tensor.permute(0, 2, 3, 1).numpy() # (B, H, W, C)
        
        plt.figure(figsize=(10, 4))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(grid[i])
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    show(real_imgs, "Immagini Reali")
    show(recon_imgs, "Ricostruzione VQ-VAE Standard")
    show(recon_from_indices, "Ricostruzione da Indici (Verità per Mamba)")

if __name__ == "__main__":
    main()