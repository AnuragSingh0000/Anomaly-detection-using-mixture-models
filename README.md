# Anomaly-detection-using-mixture-models
Implementations and extensions of DAGMM—including VAE, VQ‑VAE, Laplacian Mixture Models, and enhanced loss/inference strategies—for robust, unsupervised anomaly detection in high‑dimensional data.

# Gaussian Mixture Models for Anomaly Detection

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#requirements)

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Architecture](#architecture)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Models & Variations](#models--variations)
* [Results Summary](#results-summary)
* [Authors](#authors)
* [License](#license)

---

## Project Overview

This repository contains implementations and variations of the **Deep Autoencoding Gaussian Mixture Model (DAGMM)** for unsupervised anomaly detection. DAGMM unifies dimensionality reduction and density estimation into a single, end-to-end trainable framework, making it particularly effective in high-dimensional settings.

## Features

* **Baseline DAGMM** with reconstruction and sample energy losses
* **DAGMM–VAE**: replaces the autoencoder with a Variational Autoencoder
* **DAGMM–VQ‑VAE**: uses a Vector-Quantized VAE for discrete latent representations
* **Laplacian Mixture Models**: swaps Gaussian components for Laplace distributions
* **Modified Loss Function**: incorporates anomalous samples during training
* **Enhanced Inference Criterion**: combines reconstruction loss with sample energy
* **Threshold Estimation**: automatic threshold selection using partially labeled data

## Architecture

![DAGMM Architecture](assets/arch.png)
*Figure: Overview of the DAGMM architecture combining an autoencoder and mixture density network.*

Mathematically:

```math
p = \mathrm{MLP}(z; \theta_m), \quad \hat{\gamma} = \mathrm{softmax}(p)
```

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* [PyTorch](https://pytorch.org/) 1.7+
* NumPy, scikit-learn, matplotlib, pandas, tqdm

```bash
pip install torch numpy scikit-learn matplotlib pandas tqdm
```

### Installation

1. Clone this repository:

   ```bash
   ```

git clone [https://github.com/](https://github.com/)<your-username>/dagmm-anomaly-detection.git
cd dagmm-anomaly-detection

````
2. (Optional) Create a virtual environment:
   ```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
````

3. Install requirements:

   ```bash
   ```

pip install -r requirements.txt

````

---

## Usage

### Training
Train the baseline DAGMM:
```bash
python train.py --model dagmm --data_path data/normal.csv --epochs 100
````

Train the VAE variation:

```bash
python train.py --model dagmm_vae --data_path data/normal.csv --epochs 100
```

Other variations: `dagmm_vqvae`, `dagmm_lmm`, `dagmm_loss_mod`, `dagmm_infer_mod`, `dagmm_threshold`.

### Testing

```bash
python test.py --model checkpoints/dagmm.pth --data_path data/test.csv --threshold 0.20
```

---

## Models & Variations

| Variation                   | Key Idea                                                     |
| --------------------------- | ------------------------------------------------------------ |
| **DAGMM (Baseline)**        | Autoencoder + GMM density estimator, end-to-end training     |
| **DAGMM–VAE**               | Variational autoencoder for probabilistic latent space       |
| **DAGMM–VQ‑VAE**            | Discrete latent codes via vector quantization                |
| **DAGMM with Laplacian MM** | Laplacian mixture components instead of Gaussians            |
| **Modified Loss**           | Adds inverse energy term for anomalies during training       |
| **Modified Inference**      | Uses reconstruction error + sample energy to score anomalies |
| **Threshold Estimation**    | Automatic threshold selection using partially labeled data   |

---

## Results Summary

| Method                                | Precision  | Recall     | F1-Score   |
| ------------------------------------- | ---------- | ---------- | ---------- |
| DAGMM (Baseline)                      | 0.9297     | 0.9442     | 0.9369     |
| DAGMM with VAE                        | **0.9642** | **0.9461** | **0.9551** |
| DAGMM with VQ‑VAE                     | **0.9663** | **0.9499** | **0.9581** |
| DAGMM with Modified Inference         | 0.9505     | 0.9200     | 0.9350     |
| DAGMM with Laplacian Mixture Model    | 0.8532     | 0.7554     | 0.8014     |
| DAGMM with Modified Loss Function     | 0.4207     | 0.8556     | 0.5641     |
| DAGMM with Modified Training Setup^\* | 0.7411     | **0.9785** | 0.8434     |

> ^\* Includes partially labeled and unlabeled data for threshold estimation.

---

## Authors

* Karan Gandhi
* Anurag Singh
* Arjun Dikshit
* Abhinav Khot

**Indian Institute of Technology Gandhinagar**
CS 328: Introduction to Data Science (April 23, 2025)

---

## License

This project is licensed under the [MIT License](LICENSE).
