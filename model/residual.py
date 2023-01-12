import torch.nn as nn


class PreResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        fn_forward: nn.Module,
        **kwargs
    ) -> None:
        super().__init__()
        self.fn_forward = fn_forward
        
        default_parameters = {"kernel_size": 3, "stride": 1, "padding": 1}
        for key in kwargs:
            default_parameters[key] = kwargs[key]
        
        layers = []
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels, out_channels, **default_parameters))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, **default_parameters))
        
        self.fn_residual = nn.Sequential(*layers)
        
    def forward(self, x):
        
        residual_out = self.fn_residual(x)
        forward_out = self.fn_forward(x)
        
        return residual_out + forward_out