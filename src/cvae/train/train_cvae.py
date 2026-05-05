import sys, torch, io, os, numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL.Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_tensor
import torchvision.utils as vutils
from tensorboardX import SummaryWriter 
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURAZIONE PATH ---
EXP_NAME = "cvae_vgg_perceptual_EXAMs"
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
import config
from src.models.cvae import CelebACVAE
from src.models.perceptual_loss import VGGPerceptualLoss

# --- IPER-PARAMETRI ---
BATCH_SIZE = 128     
LATENT_DIM = 128    
LR = 2e-4            
EPOCHS = 80     
IMG_SIZE = 64     
BETA = 1.0     # Peso KL 
PERCEPTUAL_WEIGHT = 0.001 # Bilanciamento VGG Sum
ATTR_INDICES = [20, 31, 39] # Male, Smiling, Young
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot_to_tb(model, device, latent_dim):
    """
    Genera una griglia 8x4 per TensorBoard che mostra l'evoluzione 
    di tutte le combinazioni semantiche degli attributi.
    """
    model.eval()
    # Definizione delle 8 combinazioni possibili (2^3)
    scenari = [
        {"n": "F-SER-ANZ", "a": [0.,0.,0.]}, {"n": "F-SER-GIO", "a": [0.,0.,1.]},
        {"n": "F-SOR-ANZ", "a": [0.,1.,0.]}, {"n": "F-SOR-GIO", "a": [0.,1.,1.]},
        {"n": "M-SER-ANZ", "a": [1.,0.,0.]}, {"n": "M-SER-GIO", "a": [1.,0.,1.]},
        {"n": "M-SOR-ANZ", "a": [1.,1.,0.]}, {"n": "M-SOR-GIO", "a": [1.,1.,1.]}
    ]
    fig, axes = plt.subplots(8, 4, figsize=(10, 14))
    
    with torch.no_grad():
        for i, s in enumerate(scenari):
            # Trasformiamo la lista di attributi [0,1,0] in un tensore batch per il decoder
            cond = torch.tensor([s["a"]]).to(device).float().expand(4, -1)
            # Campioniamo 4 vettori casuali nello spazio latente per vedere la varietà "nello stesso stile"
            z = torch.randn(4, latent_dim).to(device)
            
            # Logica Decoder
            z_cond = torch.cat((z, cond), dim=1) # Concatenazione latente + attributi
            y = model.dec_input(z_cond).view(-1, 512, 4, 4)  # Proiezione spaziale iniziale 4x4

            # Iniezione progressiva: ad ogni upsampling iniettiamo di nuovo gli attributi
            # Questo assicura che l'attributo non venga "dimenticato" durante la ricostruzione dei dettagli
            y = model.up1(model._inject_cond(y, cond))
            y = model.up2(model._inject_cond(y, cond))
            y = model.up3(model._inject_cond(y, cond))
            y = model.up4(model._inject_cond(y, cond))
            
            for j in range(4):
                # Convertiamo da tensore [C,H,W] a immagine numpy [H,W,C] per Matplotlib
                axes[i, j].imshow(y[j].cpu().permute(1, 2, 0).numpy())
                axes[i, j].axis('off')
                if j == 0: axes[i, j].set_title(s["n"], fontsize=8, loc='left', fontweight='bold')
                
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return to_tensor(PIL.Image.open(buf).convert('RGB'))

# --- FUNZIONI DI LOSS ---
# La BCE tratta i pixel come variabili Bernoulliane: l'output Sigmoid è la probabilità
# stimata che il pixel sia acceso, il pixel reale è il target di probabilità.
reconstruction_loss_function = nn.BCELoss(reduction='sum')

def kl_loss_function(mu, log_sigma):
    """Divergenza di Kullback-Leibler calcolata come somma"""
    return 0.5 * torch.sum(mu**2 + torch.exp(2 * log_sigma) - 1 - 2 * log_sigma)

def train():
    # Setup directory e logging
    model_dir = config.MODEL_PATH / EXP_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.TRAIN_RES_PATH / EXP_NAME)

    transform = transforms.Compose([
        transforms.CenterCrop(178),           
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),               
    ])

    # Caricamento Dataset standard 
    dataset = datasets.CelebA(root=str(config.DATASET_BASIC_PATH), split='train', target_type='attr', transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Modello e Loss Percettiva (VGG 4 Slice)
    model = CelebACVAE(LATENT_DIM, len(ATTR_INDICES)).to(DEVICE)
    vgg_fn = VGGPerceptualLoss().to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Dimezza il LR ogni 20 epoche per i dettagli finali
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"--- START TRAINING: {EXP_NAME} ---")
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        for x, attr in pbar:
            # Gli attributi CelebA sono -1 (false) o 1 (true). Li mappiamo in [0, 1]
            cond = ((attr[:, ATTR_INDICES].float() + 1) / 2).to(DEVICE)
            x = x.to(DEVICE)

            optimizer.zero_grad()
            output, mu, log_sigma = model(x, cond)
            
            # Calcolo Loss 

            # BCE: Pixel-wise similarity (tende a produrre immagini sfocate)
            l_bce = reconstruction_loss_function(output, x)
            # KL: Regolarizzazione (costringe lo spazio latente a una Gaussiana)
            l_kl = kl_loss_function(mu, log_sigma)
            # VGG: Somiglianza semantica (mantiene bordi e strutture nitide)
            l_vgg = vgg_fn(output, x)
            
            # Normalizzazione 'per-sample': dividiamo per il batch size per scalare la loss 
            loss = (l_bce + (BETA * l_kl) + (PERCEPTUAL_WEIGHT * l_vgg)) / x.size(0)
            
            loss.backward()
            # Gradient clipping: previene l'esplosione dei gradienti 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.0f}"})

            # TensorBoard Logging
            if global_step % 10 == 0:
                writer.add_scalar('Loss/BCE_Sum', l_bce.item(), global_step)
                writer.add_scalar('Loss/KL_Sum', l_kl.item(), global_step)
                writer.add_scalar('Loss/VGG_Sum', (PERCEPTUAL_WEIGHT * l_vgg).item(), global_step)
                writer.add_scalar('Loss/Total_Normalized', loss.item(), global_step)

            if global_step % 1000 == 0:
                with torch.no_grad():
                    writer.add_image('Visual/Grid_8_Classes', plot_to_tb(model, DEVICE, LATENT_DIM), global_step)
            
            global_step += 1

        scheduler.step()
        # Salvataggio checkpoint epoca
        torch.save(model.state_dict(), str(model_dir / "model.pth"))
        print(f"===> Epoch {epoch} completata. Avg Loss: {epoch_loss/len(loader):.0f}")

    writer.close()
    print(f"Training concluso. Modello salvato in {model_dir}")

if __name__ == "__main__":
    # Ottimizzazione kernel convoluzionali per input di dimensione fissa
    if torch.cuda.is_available(): 
        torch.backends.cudnn.benchmark = True
    train()