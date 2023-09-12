import math
import torch


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    Parameters:
        total_steps (int): Number of epochs to reach full KL divergence weight.
        shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
    """

    def __init__(self, total_steps: int, shape: str, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        if not disable:
            self.shape = shape
        else:
            self.shape = 'none'

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.slope == 'linear':
            slope = (self.current_step -1 / self.total_steps-1)
        elif self.slope == 'cosine':
            slope = 0.5 + 0.5 * math.cos(math.pi * (self.current_step / self.total_steps))
        elif self.slope == 'logistic':
            exponent = (self.total_steps / 2) - self.current_step
            slope = 1 / (1 + math.exp(exponent))
        elif self.slope == 'none':
            slope = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        return slope

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            pass


class VAELoss(torch.nn.Module):
    """
    Calculates reconstruction loss and KL divergence loss for VAE.
    """

    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, x, x0, mu, logvar):
        """
        Args:
            x (torch.Tensor): reconstructed input tensor
            x0 (torch.Tensor): original input tensor
            mu (torch.Tensor): latent space mu
            logvar (torch.Tensor): latent space log variance
        Returns:
            bce (torch.Tensor): binary cross entropy loss (VAE recon loss)
            kld (torch.Tensor): KL divergence loss
        """
        bce = torch.nn.functional.binary_cross_entropy(x0, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce, kld
