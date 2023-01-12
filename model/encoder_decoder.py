import torch
import torch.nn as nn
from torchvision import transforms

from model.encoder import Encoder, EncoderRes
from model.decoder import Decoder
from noise_layers.noiser import Noiser
from model.discriminator import Discriminator

from model.ssim import SSIMAttenuation, psnr_clip

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = None
        self.decoder = None
        self.noiser = None
        self.ssim = None
        
        self.image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1)
        self.image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1)
        
    def forward(self, images, messages):
        encoded_images = self.encoder(images, messages)
        
        """
        if self.ssim is not None:
            encoded_images = self.ssim.apply(encoded_images, images)
        encoded_images = psnr_clip(encoded_images, images, 42., self.image_mean)
        """
        
        noised_images = self.noiser([encoded_images, images])
        decoded_messages = self.decoder(noised_images)
        return encoded_images, decoded_messages
    
    
class EncoderDecoder(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(num_blocks=config['encoder']['num_blocks'],
                               num_channels=config['encoder']['num_channels'],
                               message_length=config['message_length']).to(config['device'])

        self.decoder = Decoder(num_blocks=config['decoder']['num_blocks'],
                               num_channels=config['decoder']['num_channels'],
                               message_length=config['message_length']).to(config['device'])

        self.noiser = Noiser(config['noiser'], config['device'])
        
        self.image_mean = self.image_mean.to(config['device'])
        self.image_std = self.image_std.to(config['device'])
        
class ResEncoderDecoder(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.encoder = EncoderRes(num_blocks=config['encoder']['num_blocks'],
                               num_channels=config['encoder']['num_channels'],
                               message_length=config['message_length']).to(config['device'])

        self.decoder = Decoder(num_blocks=config['decoder']['num_blocks'],
                               num_channels=config['decoder']['num_channels'],
                               message_length=config['message_length']).to(config['device'])

        self.noiser = Noiser(config['noiser'], config['device'])
        
        self.image_mean = self.image_mean.to(config['device'])
        self.image_std = self.image_std.to(config['device'])
        
        self.ssim = SSIMAttenuation(device=config['device'])