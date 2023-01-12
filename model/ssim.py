import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable


def psnr_clip(x, y, target_psnr, image_std):
    """ 
    Clip x so that PSNR(x,y)=target_psnr 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        target_psnr: Target PSNR value in dB
    """
    delta = x - y
    delta = 255 * (delta * image_std)
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    if psnr<target_psnr:
        delta = (torch.sqrt(10**((psnr-target_psnr)/10))) * delta 
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    delta = (delta / 255.0) / image_std
    return y + delta


class SSIMAttenuation:

    def __init__(self, window_size=17, sigma=1.5, device="cpu"):
        """ Self-similarity attenuation, to make sure that the augmentations occur high-detail zones. """
        self.window_size = window_size
        _1D_window = torch.Tensor(
            [np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]
            ).to(device, non_blocking=True)
        _1D_window = (_1D_window/_1D_window.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = Variable(_2D_window.expand(3, 1, window_size, window_size).contiguous())

    def heatmap(self, img1, img2):
        """
        Compute the SSIM heatmap between 2 images, based upon https://github.com/Po-Hsun-Su/pytorch-ssim 
        Args:
            img1: Image tensor with values approx. between [-1,1]
            img2: Image tensor with values approx. between [-1,1]
            window_size: Size of the window for the SSIM computation
        """
        window = self.window
        window_size = self.window_size
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = 3)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = 3)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = 3) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = 3) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = 3) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map

    def apply(self, x, y):
        """ 
        Attenuate x using SSIM heatmap to concentrate changes of y wrt. x around edges
        Args:
            x: Image tensor with values approx. between [-1,1]
            y: Image tensor with values approx. between [-1,1], ex: original image
        """
        delta = x - y
        ssim_map = self.heatmap(x, y) # 1xCxHxW
        ssim_map = torch.sum(ssim_map, dim=1, keepdim=True)
        ssim_map = torch.clamp_min(ssim_map,0)
        delta = delta*ssim_map
        return y + delta
