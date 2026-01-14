import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file):
        """
            img_dir: Cartella dove sono le immagini 64x64
            attr_file: File txt pulito creato dallo script di preprocess
        """
        self.img_dir = img_dir
        
        # Leggiamo il file attributi
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        
        # La prima riga è il count, la seconda gli header. I dati partono dalla terza.
        self.data_lines = lines[2:]
        
        # Trasformazione: Da Immagine a Tensore [-1, 1]
        self.transform = T.Compose([
            T.ToTensor(),                 # Porta in [0, 1]
            T.Normalize((0.5, 0.5, 0.5),  # (input - 0.5) / 0.5 -> Porta in [-1, 1]
                        (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        line = self.data_lines[idx].split()
        
        # 1. Carica Immagine
        img_name = line[0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Applica trasformazioni (lo fa diventare un tensore)
        image = self.transform(image)
        
        # 2. Carica Attributi (Male, Smiling, Young)
        # Convertiamo le stringhe '0'/'1' in numeri interi
        attr = torch.tensor([int(x) for x in line[1:]], dtype=torch.float32)
        
        return image, attr