"""Quantile Temporal Convolutional Network for RUL."""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
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
    """Causal TCN → quantile RUL head."""

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
        dilations = [2 ** i for i in range(n_layers)]
        blocks = []
        in_ch = feat_dim
        for d in dilations:
            blocks.append(TCNBlock(in_ch, hidden, kernel, d, dropout))
            in_ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 3)

    @staticmethod
    def _ordered_quantiles(raw: torch.Tensor) -> torch.Tensor:
        """Enforce q05 < q50 < q95 via cumulative softplus."""
        d0 = nn.functional.softplus(raw[:, 0:1])
        d1 = nn.functional.softplus(raw[:, 1:2])
        d2 = nn.functional.softplus(raw[:, 2:3])
        q05 = d0
        q50 = q05 + d1
        q95 = q50 + d2
        return torch.cat([q05, q50, q95], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) feature sequence
        returns: (B, 3) ordered quantile RUL at final time step
        """
        h = self.tcn(x.transpose(1, 2))
        last = h[:, :, -1]
        return self._ordered_quantiles(self.head(last))


class QuantileMLP(nn.Module):
    """Single-step baseline using last feature vector only."""

    def __init__(self, feat_dim: int = 32, hidden: int = 64, n_quantiles: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x[:, -1, :])
        return QuantileTCN._ordered_quantiles(self.out(h))
