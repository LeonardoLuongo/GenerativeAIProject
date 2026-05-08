import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    """
    Loss Percettiva basata su VGG16: valuta la somiglianza semantica delle immagini 
    confrontando le mappe di caratteristiche (feature maps) a diversi livelli gerarchici.
    """
    def __init__(self):
        super().__init__()
        # Carichiamo la VGG16 pre-addestrata su ImageNet
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Dividiamo la rete in 4 fette (slices)
        # Slice 1-2: Estraggono linee, colori e texture semplici
        # Slice 3-4: Estraggono forme complesse (occhi, bocca, contorni viso)
        self.slice1 = nn.Sequential(*vgg[:4])   # conv1_2
        self.slice2 = nn.Sequential(*vgg[4:9])  # conv2_2
        self.slice3 = nn.Sequential(*vgg[9:16]) # conv3_3
        self.slice4 = nn.Sequential(*vgg[16:23])# conv4_3 
        
        # Congeliamo i pesi perché la VGG non deve imparare nulla
        for param in self.parameters(): 
            param.requires_grad = False
        self.eval()

        # Tensori di media e deviazione standard gestiti come buffer (non parametri addestrabili)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Adattiamo i pixel [0, 1] del CVAE alla distribuzione di ImageNet
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Estrazione feature ricorsiva
        h1_x, h1_y = self.slice1(input), self.slice1(target)
        h2_x, h2_y = self.slice2(h1_x), self.slice2(h1_y)
        h3_x, h3_y = self.slice3(h2_x), self.slice3(h2_y)
        h4_x, h4_y = self.slice4(h3_x), self.slice4(h3_y)
        
        # Calcoliamo la somma delle distanze al quadrato (L2) tra le mappe di caratteristiche
        # Questo penalizza le immagini che non sono 'semanticamente' simili all'originale
        loss =  torch.sum((h1_x - h1_y)**2) + \
                torch.sum((h2_x - h2_y)**2) + \
                torch.sum((h3_x - h3_y)**2) + \
                torch.sum((h4_x - h4_y)**2)
        return loss