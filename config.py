import torch

config = {
    'experiment_name': "base",
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'message_length': 30,
    'image_size': 128,
    'encoder': {'num_blocks': 7, 'num_channels': 64},
    'decoder': {'num_blocks': 4, 'num_channels': 64},
    'discriminator': {'num_blocks': 3, 'num_channels': 64},
    'noiser': [],
    'train': {'epochs': 10, 'batch_size': 32, 'train_images': None, 'val_images': None}
}