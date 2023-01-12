import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np

class Rotate(nn.Module):
    def __init__(self, rotation_range):
        super().__init__()
        self.angle_min = rotation_range[0]
        self.angle_max = rotation_range[1]

    def forward(self, noised_and_cover):
        
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        
        angle = np.random.uniform(low=self.angle_min, high=self.angle_max) 
        noised_image = F.rotate(noised_image, angle)
        return noised_image
