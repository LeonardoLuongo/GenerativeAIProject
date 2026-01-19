import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential(*vgg[:4])   # conv1_2
        self.slice2 = nn.Sequential(*vgg[4:9])  # conv2_2
        self.slice3 = nn.Sequential(*vgg[9:16]) # conv3_3
        self.slice4 = nn.Sequential(*vgg[16:23]) # conv4_3 
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        h1_x, h1_y = self.slice1(input), self.slice1(target)
        h2_x, h2_y = self.slice2(h1_x), self.slice2(h1_y)
        h3_x, h3_y = self.slice3(h2_x), self.slice3(h2_y)
        h4_x, h4_y = self.slice4(h3_x), self.slice4(h3_y)
        
        loss = torch.sum((h1_x - h1_y)**2) + \
               torch.sum((h2_x - h2_y)**2) + \
               torch.sum((h3_x - h3_y)**2) + \
               torch.sum((h4_x - h4_y)**2)
        return loss
