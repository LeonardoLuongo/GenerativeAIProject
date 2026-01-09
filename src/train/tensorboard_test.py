import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torch.utils.tensorboard import SummaryWriter # <--- Importante per TensorBoard
from config import *

# ==========================================
# 1. Configurazione e Hyperparameters
# ==========================================
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 1  # Teniamo basso per la demo
IMAGE_SIZE = 64
LATENT_DIM = 256 # Dimensione del "bottleneck"

LOG_DIR = TRAIN_RES_PATH / "celeba_mlp_demo" # Cartella dove TensorBoard salverà i dati
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando device: {DEVICE}")

# ==========================================
# 2. Dataset e Dataloader
# ==========================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Usiamo il path assoluto per sicurezza
base_path = os.getcwd()
dataset_path = os.path.join(base_path, 'dataset')

print("Caricamento Dataset...")
train_dataset = CelebA(
    root=dataset_path,
    split='train',
    target_type='attr',
    transform=transform,
    download=True 
)

dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 3. Modello: MLP Autoencoder (Semplice)
# ==========================================
# Un MLP che schiaccia l'immagine e prova a ricostruirla
class MLPAutoencoder(nn.Module):
    def __init__(self):
        super(MLPAutoencoder, self).__init__()
        # Flatten input: 3 canali * 64 * 64 = 12288
        self.input_dim = 3 * IMAGE_SIZE * IMAGE_SIZE
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, LATENT_DIM),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Sigmoid(), # Output tra 0 e 1 per le immagini
            nn.Unflatten(1, (3, IMAGE_SIZE, IMAGE_SIZE))
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

model = MLPAutoencoder().to(DEVICE)
criterion = nn.MSELoss() # Loss per la ricostruzione
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 4. Inizializzazione TensorBoard
# ==========================================
writer = SummaryWriter(LOG_DIR) 

# Aggiungiamo il GRAFO del modello (utile per il report)
# Creiamo un input fittizio per tracciare il grafo
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
writer.add_graph(model, dummy_input)

# ==========================================
# 5. Training Loop
# ==========================================
print("Inizio Training...")
step = 0

for epoch in range(EPOCHS):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(DEVICE)
        
        # Forward
        recon_images = model(real_images)
        loss = criterion(recon_images, real_images)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # === TENSORBOARD LOGGING ===
        
        # 1. Log Scalare (Loss) - Ogni 10 step
        if step % 10 == 0:
            writer.add_scalar('Training/MSE_Loss', loss.item(), step)
            
        # 2. Log Immagini (Originali vs Generate) - Ogni 100 step
        # Questo è CRUCIALE per il tuo progetto GenAI per vedere la qualità
        if step % 100 == 0:
            with torch.no_grad():
                # Creiamo una griglia: sopra immagini reali, sotto ricostruite
                real_grid = vutils.make_grid(real_images[:8], normalize=True)
                recon_grid = vutils.make_grid(recon_images[:8], normalize=True)
                
                writer.add_image('Images/Real', real_grid, step)
                writer.add_image('Images/Reconstructed', recon_grid, step)
                
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{step}] Loss: {loss.item():.4f}")

        step += 1

    # 3. Log Istogrammi dei Pesi (Fine epoca)
    # Utile per vedere se i gradienti muoiono (se la curva si schiaccia su zero)
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)

print("Training completato!")
writer.close() # Chiudere sempre il writer alla fine