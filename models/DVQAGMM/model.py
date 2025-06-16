# model_vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class VectorQuantizer(nn.Module):
    """
    VQ-VAE bottleneck layer: maintains a codebook of embeddings.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z: Tensor) -> tuple:
        # z: (B, D)
        # Compute distances to codebook
        # shape: (B, num_embeddings)
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * z @ self.embeddings.weight.t()
        )
        # get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings(encoding_indices)

        # Commitment loss
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, loss


class DAGMM_VQVAE(nn.Module):
    def __init__(self, num_embeddings=16, embedding_dim=1, n_gmm=2):
        super().__init__()
        z_dim = embedding_dim
        self.z_dim = z_dim

        # Encoder
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)

        # Estimation net
        self.fc9  = nn.Linear(z_dim + 2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, z):
        h = torch.tanh(self.fc5(z))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)

    def estimate(self, z_aug):
        h = F.dropout(torch.tanh(self.fc9(z_aug)), p=0.5, training=self.training)
        return F.softmax(self.fc10(h), dim=1)

    def compute_reconstruction(self, x, x_hat):
        rel_euc = (x - x_hat).norm(2, dim=1) / (x.norm(2, dim=1) + 1e-12)
        cos_sim = F.cosine_similarity(x, x_hat, dim=1)
        return rel_euc, cos_sim

    def forward(self, x):
        # Encoder bottleneck
        z_e = self.encode(x)
        z_q, vq_loss = self.vq_layer(z_e)
        x_hat = self.decode(z_q)

        rec1, rec2 = self.compute_reconstruction(x, x_hat)
        z_aug = torch.cat([z_q, rec1.unsqueeze(-1), rec2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z_aug)

        return {
            'z_e': z_e,
            'z_q': z_q,
            'vq_loss': vq_loss,
            'x_hat': x_hat,
            'z_aug': z_aug,
            'gamma': gamma
        }
