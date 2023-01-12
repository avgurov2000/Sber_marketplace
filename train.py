import torch
import numpy as np
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from model.discriminator import Discriminator
from validate import validate
from utils import save_model

def weights_init_discriminator(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.constant_(m.bias.data, 0)
    
def weights_init_model(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
        
def train(network, config, train_loader, val_loader):

    discriminator = Discriminator(num_blocks=config['discriminator']['num_blocks'],
                                  num_channels=config['discriminator']['num_channels']).to(config['device'])
    
    discriminator.apply(weights_init_discriminator)
    network.apply(weights_init_model)

    num_epochs = config['train']['epochs']
    msg_length = config['message_length']
    device = config['device']

    image_distortion_criterion = nn.MSELoss().to(device)
    message_distortion_criterion = nn.MSELoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

    network_optimizer = torch.optim.Adam(network.parameters(), betas=(0.85, 0.999), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    
    img_loss_arr = []
    msg_loss_arr = []
    ber_train = []
    ber_val = []
    psnr_val = []
    
    minimum_val_error = -1
    temp_epoch = np.linspace(0.03, 0.05, num_epochs)
    for epoch in tqdm(range(num_epochs)):

        image_distortion_history = []
        message_distortion_history = []
        adversarial_history = []
        ber_history = []
            
        for images, _ in train_loader:
            network.train()
            discriminator.train()

            images = images.to(device)
            messages = np.random.randint(2, size=(images.shape[0], msg_length))
            messages = torch.Tensor(messages).to(device)
            
            encoded_images, decoded_messages = network(images, messages)

            # train discriminator
            discriminator_optimizer.zero_grad()

            discriminator_prediction = discriminator(torch.cat((images.detach(),
                                                                encoded_images.detach()), 0))
            true_prediction = torch.cat((torch.full((images.shape[0], 1), 1.),
                                         torch.full((images.shape[0], 1), 0.)), 0).to(device)
            discriminator_loss = adversarial_criterion(discriminator_prediction, true_prediction)

            discriminator_loss.backward()
            discriminator_optimizer.step()

            # train encoder-decoder
            network_optimizer.zero_grad()

            messages_loss = message_distortion_criterion(messages, decoded_messages)
            messages_var_loss = (images - encoded_images).abs().var(-3).mean()
            
            images_loss = image_distortion_criterion(images, encoded_images)
            
            discriminator_prediction = discriminator(encoded_images)
            adversarial_loss_enc = adversarial_criterion(discriminator_prediction,
                                                         torch.full((images.shape[0], 1), 1.).to(device))
            ################
            ### ssl loss ###
            decoded_messages_pure = network.decoder(encoded_images.detach())
            decoded_messages_pure = nn.functional.softmax(decoded_messages_pure/torch.tensor(temp_epoch[epoch], device=device), dim=-1)
            ssl_loss = torch.sum(-decoded_messages_pure * nn.functional.log_softmax(decoded_messages/0.1, dim=-1), dim=-1).mean()
            ################
            ################
            network_loss = 0.7*images_loss + 0.001*adversarial_loss_enc + messages_loss + 0.001*ssl_loss
            network_loss.backward()
            network_optimizer.step()

            decoded_messages = torch.clip(decoded_messages.detach().round(), 0., 1.).to(device)
            ber = torch.abs(messages - decoded_messages).sum() / (images.shape[0] * msg_length)

            image_distortion_history.append(images_loss.item())
            message_distortion_history.append(messages_loss.item())
            adversarial_history.append(discriminator_loss.item())
            ber_history.append(ber.item())
            
        val_ber, val_psnr = validate(network, val_loader, msg_length, device)
        img_loss_arr.append(sum(image_distortion_history) / len(image_distortion_history))
        msg_loss_arr.append(sum(message_distortion_history) / len(message_distortion_history))
        ber_train.append(sum(ber_history) / len(ber_history))
        ber_val.append(val_ber)
        psnr_val.append(val_psnr)
        print()
        print('image_distortion:  ', img_loss_arr[-1])
        print('message_distortion:', msg_loss_arr[-1])
        print('adversarial:       ', sum(adversarial_history) / len(adversarial_history))
        print('ber:               ', ber_train[-1])

        print('validation_ber:    ', ber_val[-1])
        print('validation_psnr:    ', psnr_val[-1])
        print()
        
        if minimum_val_error == -1:
            minimum_val_error = val_ber
        elif val_ber < minimum_val_error:
                minimum_val_error = val_ber
                save_model(network, config['experiment_name'] + '_best_epoch.pth')
    return (img_loss_arr, msg_loss_arr, ber_train, ber_val, psnr_val)
