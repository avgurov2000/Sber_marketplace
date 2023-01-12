import torch
import torch.nn as nn
from model.block import Block
from model.residual import PreResidualBlock

class Encoder(nn.Module):
    
    def __init__(self, num_blocks, num_channels, message_length):
        super().__init__()
        self.num_blocks = num_blocks
        self.conv_layers = []
        self.conv_layers.append(Block(3+message_length, num_channels))
        for _ in range(1, num_blocks):
            inp_channels = num_channels
            out_channels = num_channels
            if _ % 2 == 0:
                inp_channels = num_channels+3+message_length
            self.conv_layers.append(Block(inp_channels, out_channels))
        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        self.final_layer = []
        self.final_layer.append(nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, padding=0))
        self.final_layer = nn.Sequential(*self.final_layer)
        
        
    def forward(self, image, message):

        # replicate message spatially
        message = message.unsqueeze(-1).unsqueeze(-1)
        message = message.expand(-1, -1, image.shape[-2], image.shape[-1])

        concat = torch.cat([image, message], dim=1)
        out = self.conv_layers[0](concat)
        
        for _ in range(1, self.num_blocks):
            if _ % 2 == 0:
                out = torch.cat([out, concat], dim=1)
            out = self.conv_layers[_](out)
        out = self.final_layer(out)
        return out

    
class EncoderRes(nn.Module):
    
    def __init__(self, num_blocks, num_channels, message_length):
        super().__init__()
        self.num_blocks = num_blocks
        self.conv_layers = []
        self.conv_layers.append(Block(3+message_length, num_channels))
        for _ in range(1, num_blocks):
            inp_channels = num_channels
            out_channels = num_channels
            if _ % 2 == 0:
                inp_channels = num_channels+3+message_length
                
            block = Block(inp_channels, out_channels)
            if inp_channels != out_channels:
                self.conv_layers.append(block)
            else:
                self.conv_layers.append(PreResidualBlock(inp_channels, out_channels, block))
        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        self.final_layer = []
        self.final_layer.append(nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, padding=0))
        self.final_layer = nn.Sequential(*self.final_layer)
        
        
    def forward(self, image, message):

        # replicate message spatially
        message = message.unsqueeze(-1).unsqueeze(-1)
        message = message.expand(-1, -1, image.shape[-2], image.shape[-1])

        concat = torch.cat([image, message], dim=1)
        out = self.conv_layers[0](concat)
        
        for _ in range(1, self.num_blocks):
            if _ % 2 == 0:
                out = torch.cat([out, concat], dim=1)
            out = self.conv_layers[_](out)
        out = self.final_layer(out)
        return out