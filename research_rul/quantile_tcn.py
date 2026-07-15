"""Quantile Temporal Convolutional Network for RUL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class QuantileTCN(nn.Module):
    """Causal TCN + mask-aware attention pooling → ordered quantile RUL."""

    def __init__(
        self,
        feat_dim: int = 32,
        hidden: int = 96,
        n_layers: int = 4,
        kernel: int = 3,
        quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.input_proj = nn.Conv1d(feat_dim, hidden, kernel_size=1)
        dilations = [2 ** i for i in range(n_layers)]
        blocks = []
        for d in dilations:
            blocks.append(TCNBlock(hidden, hidden, kernel, d, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        # bias toward non-degenerate intervals at init (norm space ≈ RUL/500)
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            # softplus(x)+eps ≈: q05~0.05, gap_med~0.15, gap_hi~0.25 → ~75–200 cycle band
            self.head[-1].bias.copy_(torch.tensor([-2.5, -1.5, -0.8]))

    @staticmethod
    def _ordered_quantiles(raw: torch.Tensor) -> torch.Tensor:
        """Median-centered ordered quantiles (better point estimate than lower-stacking)."""
        med = F.softplus(raw[:, 1:2])
        lo_gap = F.softplus(raw[:, 0:1]) + 1e-3
        hi_gap = F.softplus(raw[:, 2:3]) + 1e-3
        q05 = torch.relu(med - lo_gap)
        q50 = med
        q95 = med + hi_gap
        return torch.cat([q05, q50, q95], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, F)
        mask: (B, T) 1=valid
        returns: (B, 3)
        """
        h = self.input_proj(x.transpose(1, 2))
        h = self.tcn(h).transpose(1, 2)  # (B, T, H)
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        scores = self.attn(h).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask < 0.5, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        # left-padded sequences → last timestep is always the current cycle
        last = h[:, -1, :]
        feat = torch.cat([pooled, last], dim=-1)
        return self._ordered_quantiles(self.head(feat))


class QuantileMLP(nn.Module):
    """Single-step baseline using last feature vector only."""

    def __init__(self, feat_dim: int = 32, hidden: int = 128, n_quantiles: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.net(x[:, -1, :])
        return QuantileTCN._ordered_quantiles(self.out(h))
