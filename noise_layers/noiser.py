import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg import JpegCompression


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            
            self.noise_layers.append(layer)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers)
        return random_noise_layer(encoded_and_cover)