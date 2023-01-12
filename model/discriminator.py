import torch
import torch.nn as nn
from model.block import Block


class Discriminator(nn.Module):
    def __init__(self, num_blocks, num_channels):
        super().__init__()
        self.conv_layers = []
        self.conv_layers.append(Block(3, num_channels, bias=False))
        for _ in range(num_blocks - 1):
            self.conv_layers.append(Block(num_channels, num_channels, bias=False))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        self.final_step = []
        self.final_step.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.final_step.append(nn.Flatten())
        self.final_step.append(nn.Linear(num_channels, 1))
        self.final_step = nn.Sequential(*self.final_step)
        
    def forward(self, image):
        image = self.conv_layers(image)
        result = self.final_step(image)
        return result