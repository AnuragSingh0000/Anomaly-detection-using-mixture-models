import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class DAGMM_VAE(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        super(DAGMM_VAE, self).__init__()
        self.z_dim = z_dim

        # Encoder: 118→60→30→10→[μ, logσ²]
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc_mu     = nn.Linear(10, z_dim)
        self.fc_logvar = nn.Linear(10, z_dim)

        # Decoder: z→10→30→60→118
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)

        # Estimation net: (z + rec1 + rec2) → 10 → n_gmm
        self.fc9  = nn.Linear(z_dim + 2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.tanh(self.fc5(z))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)

    def estimate(self, z_aug):
        h = F.dropout(torch.tanh(self.fc9(z_aug)), p=0.5, training=self.training)
        return F.softmax(self.fc10(h), dim=1)

    def compute_reconstruction(self, x, x_hat):
        # same two summary metrics
        rel_euc = (x - x_hat).norm(2, dim=1) / (x.norm(2, dim=1) + 1e-12)
        cos_sim = F.cosine_similarity(x, x_hat, dim=1)
        return rel_euc, cos_sim

    def forward(self, x):
        # VAE encode → sample → decode
        mu, logvar = self.encode(x)
        z_c = self.reparameterize(mu, logvar)
        x_hat = self.decode(z_c)

        # reconstruction summaries
        rec1, rec2 = self.compute_reconstruction(x, x_hat)

        # concat for GMM
        z_aug = torch.cat([
            z_c,
            rec1.unsqueeze(-1),
            rec2.unsqueeze(-1)
        ], dim=1)

        gamma = self.estimate(z_aug)
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z_c,
            'x_hat': x_hat,
            'z_aug': z_aug,
            'gamma': gamma
        }
