
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf

class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device

    def forward(self, x, x_hat, z, gamma):
        # Reconstruction loss (MSE)
        reconst_loss = torch.mean((x - x_hat).pow(2))
        # Compute mixture parameters and sample energy
        phi, mu, b = self.compute_params(z, gamma)
        sample_energy = self.compute_energy(z, gamma, phi, mu, b)
        # Regularization on scale parameters to prevent negative energies
        eps = 1e-12
        scale_reg = torch.sum(1.0 / (b + eps))
        # Total loss
        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * scale_reg
        return Variable(loss, requires_grad=True)

    def compute_params(self, z, gamma):
        # phi_k: mixture weights
        phi = torch.sum(gamma, dim=0) / gamma.size(0)
        # mu_k: component means
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu = mu / torch.sum(gamma, dim=0).unsqueeze(-1)

        # b_k: scale (mean absolute deviation)
        N_k = torch.sum(gamma, dim=0)  # K
        D = z.size(1)
        diff = torch.abs(z.unsqueeze(1) - mu.unsqueeze(0))  # N x K x D
        b_num = torch.sum(gamma.unsqueeze(-1) * diff, dim=(0, 2))  # K
        b = b_num / (N_k * D + 1e-12)
        return phi, mu, b

    def compute_energy(self, z, gamma, phi=None, mu=None, b=None, sample_mean=True):
        # Compute sample energy under Laplacian mixture
        if phi is None or mu is None or b is None:
            phi, mu, b = self.compute_params(z, gamma)

        N, D = z.size()
        # Compute L1 distances N x K
        dist = torch.sum(torch.abs(z.unsqueeze(1) - mu.unsqueeze(0)), dim=2)
        # Log-coeff and exponent
        eps = 1e-12
        log_coef = -D * torch.log(2 * b + eps)  # K
        exponent = -dist / (b.unsqueeze(0) + eps) + log_coef.unsqueeze(0)  # N x K

        # Mixture log-likelihood
        weighted = phi.unsqueeze(0) * torch.exp(exponent)
        E = -torch.log(torch.sum(weighted, dim=1) + eps)
        if sample_mean:
            E = torch.mean(E)
        return E