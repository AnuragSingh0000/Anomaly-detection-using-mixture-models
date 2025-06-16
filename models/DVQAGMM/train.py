# train_vqvae.py
import torch
from torch import optim
from barbar import Bar

from model import DAGMM_VQVAE
from forward_step import ComputeLossVQVAE
from utils.utils import weights_init_normal

class TrainerDAGMMVQVAE:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def train(self):
        self.model = DAGMM_VQVAE(
            num_embeddings=self.args.num_embeddings,
            embedding_dim=self.args.latent_dim,
            n_gmm=self.args.n_gmm
        ).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        compute = ComputeLossVQVAE(
            lambda_energy=self.args.lambda_energy,
            lambda_cov=self.args.lambda_cov,
            device=self.device,
            n_gmm=self.args.n_gmm
        )

        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = compute.forward(x, outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch} VQ-VAE DAGMM Loss: {total_loss/len(self.train_loader):.3f}")
