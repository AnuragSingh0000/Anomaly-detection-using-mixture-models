import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

from forward_step import ComputeLoss


def eval(model, dataloaders, device, n_lmm, sub=20):
    """Evaluate DALMM with varying contamination threshold sub (percentage)."""
    train_loader, test_loader = dataloaders
    model.eval()
    compute = ComputeLoss(model, lambda_energy=1.0, lambda_cov=0.0, device=device)

    # Estimate global mixture parameters from train set
    with torch.no_grad():
        gamma_sum = 0
        mu_sum = 0
        b_sum = 0
        N_samples = 0
        for x, _ in train_loader:
            x = x.float().to(device)
            _, _, z, gamma = model(x)
            phi_batch, mu_batch, b_batch = compute.compute_params(z, gamma)
            batch_gamma = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma
            mu_sum += mu_batch * batch_gamma.unsqueeze(-1)
            b_sum += b_batch * batch_gamma
            N_samples += x.size(0)

        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_b = b_sum / gamma_sum

        # Helper to compute energies and labels
        def get_energy(loader):
            energies = []
            labels = []
            for x, y in loader:
                x = x.float().to(device)
                _, _, z, gamma = model(x)
                E = compute.compute_energy(z, gamma, train_phi, train_mu, train_b, sample_mean=False)
                energies.append(E.detach().cpu().numpy())
                labels.append(y.numpy())
            return np.concatenate(energies), np.concatenate(labels)

        energy_train, labels_train = get_energy(train_loader)
        energy_test, labels_test = get_energy(test_loader)

    # Combine for thresholding
    scores_total = np.concatenate((energy_train, energy_test))
    labels_total = np.concatenate((labels_train, labels_test))

    # Determine threshold from contamination sub (%)
    threshold = np.percentile(scores_total, 100 - sub)
    preds = (energy_test > threshold).astype(int)

    precision, recall, f1, _ = prf(labels_test, preds, average='binary')
    acc = accuracy_score(labels_test, preds)
    roc = roc_auc_score(labels_total, scores_total)

    print(f"Contamination: {sub}% | Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1: {f1:.4f}, Acc: {acc:.4f}, ROC AUC: {roc*100:.2f}")
    return labels_total, scores_total