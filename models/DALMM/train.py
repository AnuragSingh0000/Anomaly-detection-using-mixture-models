import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import DALMM
from forward_step import ComputeLoss
from utils.utils import weights_init_normal

class TrainerDALMM:
    """Trainer class for DALMM using Laplacian Mixture Model"""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        # initialize model
        self.model = DALMM(n_lmm=args.n_lmm, z_dim=args.latent_dim).to(device)
        # loss with both energy and scale regularization
        self.loss_fn = ComputeLoss(self.model,
                                   lambda_energy=args.lambda_energy,
                                   lambda_cov=args.lambda_cov,
                                   device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_milestones,
            gamma=0.1
        )
        # initialize weights if needed
        self.model.apply(weights_init_normal)

    def train(self):
        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, self.args.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            total_samples = 0

            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                self.optimizer.zero_grad()
                z_c, x_hat, z, gamma = self.model(x)
                loss = self.loss_fn.forward(x, x_hat, z, gamma)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

            epoch_loss = running_loss / total_samples
            self.scheduler.step()
            print(f"Epoch {epoch}/{self.args.num_epochs} - Loss: {epoch_loss:.6f}")

            # Early stopping & checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), 'best_dalmm.pth')
            elif epoch - best_epoch >= self.args.patience:
                print(f"Stopping early at epoch {epoch}")
                break

        # load best model weights
        self.model.load_state_dict(torch.load('best_dalmm.pth', map_location=self.device))
        print(f"Training complete. Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")
        return self.model
    
