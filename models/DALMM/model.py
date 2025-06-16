import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf



class DALMM(nn.Module):
    def __init__(self, n_lmm=2, z_dim=1):
        """Network for DALMM (KDDCup99) with Laplacian Mixture Model"""
        super(DALMM, self).__init__()
        # Encoder network
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)

        # Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)

        # Estimation network
        # latent dim + 2 reconstruction metrics
        self.fc9 = nn.Linear(z_dim + 2, 10)
        self.fc10 = nn.Linear(10, n_lmm)

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

    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)

    def compute_reconstruction(self, x, x_hat):
        rel_euc = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cos_sim = F.cosine_similarity(x, x_hat, dim=1)
        return rel_euc, cos_sim

    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_euc, rec_cos = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_euc.unsqueeze(-1), rec_cos.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma

