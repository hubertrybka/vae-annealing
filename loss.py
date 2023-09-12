import torch


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
