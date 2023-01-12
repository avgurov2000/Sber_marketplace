import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np

class Hflip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, noised_and_cover):
        
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        
        noised_image = F.hflip(noised_image)
        return noised_image
