import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import v2
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys

# --- SETUP PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from model.vq_vae.vq_vae import VQVAE

# --- CONFIGURAZIONE (Deve matchare il training) ---
BATCH_SIZE = 256
EMBEDDING_DIMENSION = 1024 
LATENT_DIM = 256           
CHECKPOINT_PATH = "checkpoints/vqvae_16x16_2_epoch_50.pth" # VERIFICA IL NOME DEL FILE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- UTILS METRICS ---
class SobelLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.filter_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.filter_x = self.filter_x.repeat(3, 1, 1, 1)
        self.filter_y = self.filter_y.repeat(3, 1, 1, 1)

    def forward(self, pred, target):
        pred_dx = torch.nn.functional.conv2d(pred, self.filter_x, padding=1, groups=3)
        pred_dy = torch.nn.functional.conv2d(pred, self.filter_y, padding=1, groups=3)
        pred_grad = torch.sqrt(pred_dx**2 + pred_dy**2 + 1e-6)

        target_dx = torch.nn.functional.conv2d(target, self.filter_x, padding=1, groups=3)
        target_dy = torch.nn.functional.conv2d(target, self.filter_y, padding=1, groups=3)
        target_grad = torch.sqrt(target_dx**2 + target_dy**2 + 1e-6)

        return torch.nn.functional.mse_loss(pred_grad, target_grad)

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def denorm(img_tensor):
    return img_tensor * 0.5 + 0.5

def evaluate():
    print(f"--- INIZIO VALUTAZIONE SU {DEVICE} ---")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # 1. SETUP DATASET
    transform = v2.Compose([
        v2.CenterCrop(178),
        v2.Resize((64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        val_set = CelebA(root='./dataset', split='valid', target_type='attr', transform=transform, download=True)
    except RuntimeError as e:
        print(f"Errore dataset: {e}")
        return

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. CARICAMENTO MODELLO
    model = VQVAE(
        num_embeddings=EMBEDDING_DIMENSION,
        embedding_dim=LATENT_DIM
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modello caricato dall'epoca {checkpoint['epoch']}")
    else:
        print(f"ERRORE: Checkpoint {CHECKPOINT_PATH} non trovato!")
        return

    model.eval()
    
    # 3. METRICHE
    mse_fn = nn.MSELoss()
    l1_fn = nn.L1Loss()
    sobel_fn = SobelLoss(DEVICE)
    
    metrics = {
        'mse': 0.0,
        'l1': 0.0,
        'sobel': 0.0,
        'perplexity': 0.0,
        'psnr': 0.0
    }
    
    codebook_usage_count = torch.zeros(EMBEDDING_DIMENSION).to(DEVICE)
    writer = SummaryWriter("runs/vqvae_evaluation")

    # 4. LOOP VALUTAZIONE
    loop = tqdm.tqdm(val_loader, leave=True)
    
    with torch.no_grad():
        for i, (x, _) in enumerate(loop):
            x = x.to(DEVICE)
            
            # Forward Standard
            x_recon, _, perp = model(x)
            
            # --- LOGICA MAMBA ---
            # 1. Estrai gli indici (quello che farà l'encoder -> mamba)
            indices = model.get_codebook_indices(x) 
            
            # 2. Statistiche Codebook
            unique_indices = torch.unique(indices)
            codebook_usage_count[unique_indices] = 1 
            
            # 3. Loss Calculation
            metrics['mse'] += mse_fn(x_recon, x).item()
            metrics['l1'] += l1_fn(x_recon, x).item()
            metrics['sobel'] += sobel_fn(x_recon, x).item()
            metrics['perplexity'] += perp.item()
            metrics['psnr'] += calculate_psnr(x_recon, x).item()

            # --- SALVATAGGIO IMMAGINI (Solo primo batch) ---
            if i == 0:
                fixed_x = x[:16]           # Originali
                fixed_recon = x_recon[:16] # Ricostruzione Standard (con gradiente passante)
                
                # Ricostruzione ESPLICITA dagli indici (Simulazione Mamba)
                fixed_recon_indices = model.decode_from_indices(indices[:16])

    # 5. AGGREGAZIONE
    num_batches = len(val_loader)
    final_metrics = {k: v / num_batches for k, v in metrics.items()}
    
    total_codes_used = torch.sum(codebook_usage_count).item()
    usage_percent = (total_codes_used / EMBEDDING_DIMENSION) * 100

    print("\n" + "="*30)
    print("   RISULTATI VALIDAZIONE")
    print("="*30)
    print(f"MSE Loss:              {final_metrics['mse']:.5f}")
    print(f"PSNR:                  {final_metrics['psnr']:.2f} dB")
    print(f"Sobel Loss:            {final_metrics['sobel']:.5f}")
    print("-" * 30)
    print(f"Codebook Usage:        {int(total_codes_used)}/{EMBEDDING_DIMENSION} ({usage_percent:.1f}%)")
    print("="*30)

    # 6. TENSORBOARD LOGGING
    writer.add_hparams({"model": "VQVAE-16x16"}, final_metrics)
    
    # Preparazione Griglie
    grid_x = torchvision.utils.make_grid(denorm(fixed_x.cpu()), nrow=4)
    grid_recon = torchvision.utils.make_grid(denorm(fixed_recon.cpu()), nrow=4)
    grid_indices = torchvision.utils.make_grid(denorm(fixed_recon_indices.cpu()), nrow=4)
    
    # --- FINESTRA 1: Standard Check ---
    # Confronto Originale vs Ricostruzione VQ-VAE standard
    combined_standard = torch.cat([grid_x, grid_recon], dim=1)
    writer.add_image("Validation/Original_vs_Standard", combined_standard, 0)
    
    # --- FINESTRA 2: Mamba Check (QUELLA CHE HAI CHIESTO) ---
    # Confronto Originale vs Ricostruzione da Indici
    # Questa è la "Ground Truth" per Mamba. Se questa fa schifo, Mamba farà schifo.
    combined_mamba = torch.cat([grid_x, grid_indices], dim=1)
    writer.add_image("Validation/Mamba_Simulation_From_Indices", combined_mamba, 0)
    
    print(f"\nLog salvati. Controlla su Tensorboard la tab 'Validation/Mamba_Simulation_From_Indices'")
    writer.close()

if __name__ == "__main__":
    evaluate()