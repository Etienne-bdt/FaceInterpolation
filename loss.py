import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, gamma):
    # Define the loss function for the VAE
    # Gamma is the variance of the prior
    b,c,h,w = x.size()
    n_x = b * c * h * w
    mse = n_x * (
        F.mse_loss(recon_x, x, reduction="mean") / (2 * gamma.pow(2)) + (gamma.log())
    )
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    return mse, kld
