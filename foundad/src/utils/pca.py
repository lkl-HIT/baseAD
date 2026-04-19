"""Lightweight GPU-accelerated PCA for patch-level anomaly scoring."""

import logging
import torch

logger = logging.getLogger(__name__)


class PCAScorer:
    """Fit PCA on normal patch features; score test patches by reconstruction residual."""

    def __init__(self, ev_ratio: float = 0.99, device=None):
        self.ev_ratio = ev_ratio
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = None
        self.C = None
        self.k = None

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "PCAScorer":
        """
        Fit PCA on normal patch features.

        Parameters
        ----------
        features : Tensor [N, D]
            All normal patch features collected from training images.
        """
        X = features.to(self.device, dtype=torch.float64)
        N, D = X.shape
        self.mu = X.mean(dim=0)
        Xc = X - self.mu
        cov = Xc.T @ Xc / max(N - 1, 1)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        cumvar = eigvals.cumsum(0) / eigvals.sum()
        self.k = min(
            int(
                torch.searchsorted(
                    cumvar,
                    torch.tensor(self.ev_ratio, dtype=torch.float64, device=self.device),
                ).item()
                + 1
            ),
            D,
        )
        self.C = eigvecs[:, : self.k]
        logger.info(
            "PCA: k=%d components, %.1f%% variance (N=%d, D=%d)",
            self.k, self.ev_ratio * 100, N, D,
        )
        return self

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute squared reconstruction residual as anomaly score.

        Parameters
        ----------
        features : Tensor [B, P, D] or [N, D]

        Returns
        -------
        Tensor [B, P] or [N] — per-patch anomaly score.
        """
        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        B, P, D = features.shape
        x = features.reshape(-1, D).to(self.device, dtype=torch.float64)
        xc = x - self.mu
        proj = xc @ self.C @ self.C.T + self.mu
        res = (x - proj).pow(2).sum(-1).float()
        return res.reshape(B, P).squeeze(0) if squeeze else res.reshape(B, P)
