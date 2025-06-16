import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf

from forward_step import ComputeLossVAE
from model import DAGMM_VAE

def eval(model, dataloaders, device, n_gmm):
    """Evaluate the DAGMM-VAE model with GMM energy scoring."""
    train_loader, test_loader = dataloaders
    model.eval()
    print('Evaluating DAGMM-VAE...')

    # Use ComputeLossVAE only for its compute_params and compute_energy methods
    compute = ComputeLossVAE(
        lambda_energy=0.0,
        lambda_cov=0.0,
        lambda_kl=0.0,
        device=device,
        n_gmm=n_gmm
    )

    # 1) Estimate GMM parameters on training (clean) data
    with torch.no_grad():
        N = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        for x, _ in train_loader:
            x = x.float().to(device)
            outputs = model(x)
            z = outputs['z']
            gamma = outputs['gamma']

            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)
            batch_gamma_sum = gamma.sum(dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum    += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum   += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            N += x.size(0)

        phi = gamma_sum / N
        mu  = mu_sum / gamma_sum.unsqueeze(-1)
        cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

    # 2) Compute energy scores for train and test
    def get_scores(loader):
        scores, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.float().to(device)
                outputs = model(x)
                z = outputs['z']
                gamma = outputs['gamma']
                energy, _ = compute.compute_energy(z, gamma, phi=phi, mu=mu, cov=cov, sample_mean=False)
                scores.append(energy.cpu())
                labels.append(y)
        return torch.cat(scores).numpy(), torch.cat(labels).numpy()

    energy_train, labels_train = get_scores(train_loader)
    energy_test,  labels_test  = get_scores(test_loader)

    # Combine for threshold and AUC
    all_scores = np.concatenate([energy_train, energy_test])
    all_labels = np.concatenate([labels_train,  labels_test])

    # Set threshold (e.g., top 20% anomalies)
    thresh = np.percentile(all_scores, 80)
    preds = (energy_test > thresh).astype(int)

    precision, recall, f1, _ = prf(labels_test, preds, average='binary')
    auc = roc_auc_score(all_labels, all_scores)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"ROC AUC: {auc*100:.2f}%")

    return all_labels, all_scores
