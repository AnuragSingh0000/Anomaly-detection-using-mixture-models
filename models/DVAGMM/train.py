import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import DAGMM_VAE
from forward_step import ComputeLossVAE
from utils.utils import weights_init_normal

class TrainerDAGMMVAE:
    """Trainer class for DAGMM with VAE."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def train(self):
        """Training the DAGMM-VAE model"""
        # Initialize model and loss
        self.model = DAGMM_VAE(self.args.n_gmm, self.args.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLossVAE(
            lambda_energy=self.args.lambda_energy,
            lambda_cov=self.args.lambda_cov,
            lambda_kl=self.args.lambda_kl,
            device=self.device,
            n_gmm=self.args.n_gmm
        )

        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x)

                # Compute loss
                loss = self.compute.forward(x, outputs)
                loss.backward()

                # Gradient clipping and optimization
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f'Training DAGMM-VAE... Epoch: {epoch}, Loss: {avg_loss:.3f}')