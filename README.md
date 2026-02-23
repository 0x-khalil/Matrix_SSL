# Matrix-SSL: Self-Supervised Learning from Scratch

An implementation of **Matrix-SSL** (Matrix Information Theory for SSL) applied to the CIFAR-10 dataset. This project explores feature discovery without labels by minimizing redundancy in the cross-correlation matrix of augmented views.

[![Blog Post](https://img.shields.io/badge/Blog-Read_Technical_Writeup-yellow?style=for-the-badge&logo=medium)](https://khalilbnrblog.vercel.app/SSL_Matrix.html) 

## 🚀 Overview
The core objective is to train a ResNet-18 encoder to learn robust visual representations using the principle of **Redundancy Reduction**. Instead of contrasting individual samples, we force the cross-correlation matrix of twin embeddings toward the identity matrix.

### Key Features
- **Projector Head:** 3-layer MLP with 2048 hidden units.
- **Optimization:** Adam optimizer with Cosine Annealing and Linear Warmup.
- **Hardware Optimized:** Leverages Mixed Precision (FP16) and Gradient Accumulation to run effectively on 8GB VRAM (NVIDIA RTX 3070 Laptop).

## 📊 Performance
- **KNN Top-1 Accuracy:** 34.92% (CIFAR-10)
- **Latent Space:** 128-dimensional embeddings.

The model experiences a "reorganization phase" between epochs 20-50 where accuracy temporarily dips as it transitions from low-level color features to more complex semantic structures.

## 🛠️ Tech Stack
- **Framework:** PyTorch
- **Visualizations:** Matplotlib, Scikit-learn (t-SNE)
- **Hardware:** RTX 3070 Laptop GPU

## 📂 Project Structure
- `model.py`: ResNet-18 backbone and MLP projector architecture.
- `loss.py`: Implementation of Matrix-Uniformity and Alignment loss.
- `plot.py`: t-SNE visualization script.
- `data_utils.py`: CIFAR-10 loading and SSL augmentation pipeline.

## 📖 Primary Inspiration
This project is based on the theoretical framework presented in:
> **Zhang et al. (2023).** *Matrix Information Theory for Self-Supervised Learning.* [arXiv:2305.17326](https://arxiv.org/abs/2305.17326)

## 🔗 Blog & Writeup
I wrote a detailed post about the "Valley of Despair" during training and the math behind the redundancy reduction on my personal website:
[Exploring the Latent Space: Matrix-SSL from Scratch](https://khalilbnr.github.io/blog/matrix-ssl)
