import torch
import torch.nn as nn
from model.block import Block
from model.residual import PreResidualBlock


class Decoder(nn.Module):
    def __init__(self, num_blocks, num_channels, message_length):
        super().__init__()
        self.conv_layers = []
        self.conv_layers.append(Block(3, num_channels))
        for _ in range(num_blocks - 1):
            self.conv_layers.append(Block(num_channels, num_channels))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        self.final_step = []
        self.final_step.append(Block(num_channels, message_length))
        self.final_step.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.final_step.append(nn.Flatten())
        self.final_step.append(nn.Linear(message_length, message_length))
        self.final_step = nn.Sequential(*self.final_step)
        
    def forward(self, encoded_image):
        encoded_image = self.conv_layers(encoded_image)
        message = self.final_step(encoded_image)

        return message
    
    
