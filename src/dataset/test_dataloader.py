import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from config import * 

# 1. Definiamo le trasformazioni
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ridimensiona
    transforms.CenterCrop((64, 64)), # Ritaglia al centro per avere solo la faccia
    transforms.ToTensor(),        # Converte in tensore PyTorch
])

# 2. Scarichiamo il dataset
# Verrà creata una cartella 'data' nella directory dove ti trovi
train_dataset = CelebA(
    root='./dataset',       # Cartella dove salvare i dati
    split='train',       # 'train', 'valid', o 'test'
    target_type='attr',  # Ci servono gli attributi (Smiling, Male, ecc.) 
    transform=transform,
    download=False       
)

# Test veloce per vedere se carica
loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
images, attributes = next(iter(loader))
print(f"Batch shape: {images.shape}") 
# Deve stampare torch.Size([4, 3, 64, 64])