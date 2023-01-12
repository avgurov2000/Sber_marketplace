import torch
from torch import nn

import torchvision
from torchvision import transforms
import numpy as np

from utils import PeakSignalNoiseRatio 



def validate(model, val_loader, msg_length, device):
    ber_history = []
    psnr_history = []
    
    fn_psnr = PeakSignalNoiseRatio()
    fn_psnr.to(device)
    with torch.no_grad():
        for images, _ in val_loader:

            model.eval()

            messages = np.random.randint(2, size=(images.shape[0], msg_length))
            messages = torch.Tensor(messages).to(device)
            images = images.to(device)

            encoded_images, decoded_messages = model(images, messages)

            decoded_messages = torch.clip(decoded_messages.detach().round(), 0., 1.).to(device)
            ber = torch.abs(messages - decoded_messages).sum() / (images.shape[0] * msg_length)
            ber_history.append(ber.item())
            
            psnr = fn_psnr(images, encoded_images)
            psnr_history.append(psnr.item())
            
    return sum(ber_history) / len(ber_history), sum(psnr_history) / len(psnr_history)

