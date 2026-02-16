import torch
import torch.nn as nn

class MatrixSSLLoss(nn.Module):
    def __init__(self, lambd=5e-3, eps=1e-6):
        super().__init__()
        self.lambd = lambd # Penalty for redundancy
        self.eps = eps

    def compute_covariance(self, z):
            # Normalize along the batch dimension to prevent scale explosion
            z = (z - z.mean(0)) / (z.std(0) + self.eps)

            batch_size = z.size(0)
            # Covariance matrix C: [Dim, Dim]
            cov = (z.T @ z) / (batch_size - 1)
            return cov

    def log_det(self, matrix):
        """
        Approximates the log-determinant using eigenvalues.
        LogDet(M) = sum(log(eigenvalues))
        """
        # We add a small identity matrix for numerical stability (regularization)
        iden = torch.eye(matrix.size(0), device=matrix.device)
        reg_matrix = matrix + self.eps * iden

        # In a real implementation, we use SVD or Cholesky for speed
        # Here we use eigenvalues for clarity in your blog post logic
        eigenvalues = torch.linalg.eigvalsh(reg_matrix)
        return torch.log(eigenvalues.clamp(min=self.eps)).sum()

    def forward(self, z1, z2):
            # 1. Compute Covariance for both augmented views
            c1 = self.compute_covariance(z1)
            c2 = self.compute_covariance(z2)

            # 2. Invariance Loss (Alignment)
            # We want the cross-correlation between views to be high
            # (This is essentially making the views look similar)
            loss_inv = torch.dist(z1, z2, p=2).pow(2) / z1.size(0)

            # 3. Redundancy Reduction (The "Matrix" magic)
            # We want the off-diagonal elements of the covariance to be 0 (Identity)
            # This forces every feature to learn something UNIQUE.
            off_diagonal_mask = ~torch.eye(c1.size(0), device=c1.device).bool()

            loss_red = (c1[off_diagonal_mask].pow(2).sum() +
                        c2[off_diagonal_mask].pow(2).sum())

            # Total Loss = Alignment + Lambda * Redundancy
            return loss_inv + self.lambd * loss_red
