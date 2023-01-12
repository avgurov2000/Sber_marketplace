from PIL import Image
import glob

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from pathlib import Path

def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


class PeakSignalNoiseRatio(nn.Module):
    def __init__(
        self,
        base: int = 255,
        un_normalize: object = None
    ):
        super().__init__()
        self.base = base
        if un_normalize is None:
            self.un_normalize = transforms.Compose([
                transforms.Normalize((0, 0, 0), (1/0.229, 1/0.224, 1/0.225)),
                transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1))
            ])
        else:
            self.un_normalize = un_normalize
            
        self.mse_loss = nn.MSELoss()
            
    def forward(self, x, y):
        x, y = self.un_normalize(x)*self.base, self.un_normalize(y)*self.base
        mse = self.mse_loss(x, y)
        if mse == 0:
            return 100
        else:
            return 20*torch.log10(self.base/mse.sqrt())