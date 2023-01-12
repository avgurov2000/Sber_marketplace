import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, **kwargs):
        super().__init__()
        default_parameters = {"kernel_size": 3, "stride": 1, "padding": 1}
        for key in kwargs:
            default_parameters[key] = kwargs[key]
            
        self.encode = []
        self.encode.append(nn.Conv2d(in_channels, out_channels, **default_parameters))
        if bn: self.encode.append(nn.BatchNorm2d(out_channels))
        self.encode.append(nn.ReLU())
        
        self.encode = nn.Sequential(*self.encode)

    def forward(self, x):
        return self.encode(x)