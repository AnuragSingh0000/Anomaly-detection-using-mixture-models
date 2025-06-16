# forward_step_vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ComputeLossVQVAE:
    def __init__(self, lambda_energy, lambda_cov, device, n_gmm):
        self.lambda_energy = lambda_energy
        self.lambda_cov    = lambda_cov
        self.device        = device
        self.n_gmm         = n_gmm

    def forward(self, x, outputs):
        x_hat  = outputs['x_hat']
        z_q    = outputs['z_q']
        vq_loss= outputs['vq_loss']
        gamma  = outputs['gamma']

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        # VQ commitment loss
        commitment_loss = vq_loss.mean()

        # GMM energy & cov penalty
        sample_energy, cov_diag = self.compute_energy(z_q, gamma)

        loss = recon_loss + commitment_loss + \
               self.lambda_energy * sample_energy + \
               self.lambda_cov    * cov_diag
        return loss

    # compute_energy and compute_params identical to ComputeLossVAE above
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        if phi is None or mu is None or cov is None:
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        eps = 1e-12

        cov_inverse = []
        det_cov     = []
        cov_diag    = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + torch.eye(cov[k].size(-1), device=self.device) * eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            L = torch.cholesky(cov_k, upper=False)
            det_cov.append((torch.diag(L).prod() * (2*np.pi)**(cov_k.size(-1)/2)).unsqueeze(0))
            cov_diag += torch.sum(1.0 / cov_k.diag())
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov     = torch.cat(det_cov).to(self.device)

        Ez = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu,
            dim=-1
        )
        Ez = torch.exp(Ez)
        probs = phi.unsqueeze(0) * Ez / torch.sqrt(det_cov).unsqueeze(0)
        sample_energy = -torch.log(torch.sum(probs, dim=1) + eps)
        if sample_mean:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def compute_params(self, z, gamma):
        phi = torch.sum(gamma, dim=0) / gamma.size(0)
        mu  = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0) / torch.sum(gamma, dim=0).unsqueeze(-1)
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov = cov / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
        return phi, mu, cov
