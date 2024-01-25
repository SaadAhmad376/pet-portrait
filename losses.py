import torch
import torch.nn.functional as F


def psnr_loss(recon, target):
    mse = F.mse_loss(recon, target, reduction='mean')
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr

def pixel_to_pixel_loss(recon, target):
    return F.l1_loss(recon, target, reduction='mean')