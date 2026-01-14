import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from tqdm import tqdm

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.vq_vae.vq_vae import VQVAE

# CONFIG
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH_VQ = "checkpoints/vqvae_16x16_2_epoch_50.pth" # IL TUO CHECKPOINT VQ-VAE
OUTPUT_FILE = "dataset/celeba_tokens.pt"

def main():
    print(f"--- PREPARAZIONE DATASET TOKENIZZATO ---")
    
    # 1. Carica VQ-VAE
    vqvae = VQVAE(num_embeddings=1024, embedding_dim=256).to(DEVICE)
    vqvae.load_state_dict(torch.load(CHECKPOINT_PATH_VQ, map_location=DEVICE)['model_state_dict'])
    vqvae.eval()
    
    # 2. Dataset Immagini
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CelebA(root='./dataset', split='train', target_type='attr', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    all_indices = []
    all_attrs = []
    
    print("Inizio estrazione token...")
    with torch.no_grad():
        for imgs, attrs in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            
            # Estraiamo solo gli indici (Interi)
            # Input: (B, 3, 64, 64) -> Output: (B, 256)
            indices = vqvae.get_codebook_indices(imgs)
            
            # Salviamo su CPU per non intasare la GPU
            all_indices.append(indices.cpu())
            # Salviamo attributi specifici (Male, Smiling, Young)
            all_attrs.append(attrs[:, [20, 31, 39]].long().cpu())
            
    # Concateniamo tutto
    full_indices = torch.cat(all_indices, dim=0).to(dtype=torch.int16) # int16 basta per 1024 valori
    full_attrs = torch.cat(all_attrs, dim=0).to(dtype=torch.int8)
    
    print(f"\nSalvataggio in {OUTPUT_FILE}...")
    torch.save({
        'indices': full_indices,
        'attrs': full_attrs
    }, OUTPUT_FILE)
    
    print("Fatto! Ora il dataset è piccolissimo e velocissimo da caricare.")

if __name__ == "__main__":
    main()