"""Pinball loss and monotonic RUL penalty."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    pred: (B, Q), target: (B,), quantiles: (Q,)
    Optional per-quantile weights to emphasize interval extremes.
    """
    target = target.unsqueeze(-1)
    errors = target - pred
    q = quantiles.view(1, -1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    if weights is not None:
        loss = loss * weights.view(1, -1)
        return loss.sum(dim=-1).mean()
    return loss.mean()


def monotonic_penalty_per_cell(
    pred_median: torch.Tensor,
    cell_ids: list[str],
    cycles: torch.Tensor,
) -> torch.Tensor:
    """Penalize RUL increases along cycle order within each cell in a batch."""
    if pred_median.numel() < 2:
        return pred_median.new_tensor(0.0)
    total = pred_median.new_tensor(0.0)
    count = 0
    cycles_cpu = cycles.detach().cpu().numpy()
    for cell in set(cell_ids):
        idx = [i for i, c in enumerate(cell_ids) if c == cell]
        if len(idx) < 2:
            continue
        order = sorted(idx, key=lambda i: cycles_cpu[i])
        seq = pred_median[order]
        diff = seq[1:] - seq[:-1]
        total = total + torch.relu(diff).pow(2).sum()
        count += len(diff)
    return total / max(count, 1)


class QuantileRULLoss(nn.Module):
    def __init__(
        self,
        quantiles=(0.05, 0.5, 0.95),
        lambda_mono: float = 0.05,
        quantile_weights=(2.0, 1.0, 2.0),
    ):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer(
            "quantile_weights", torch.tensor(quantile_weights, dtype=torch.float32)
        )
        self.lambda_mono = lambda_mono

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cell_ids: list[str] | None = None,
        cycles: torch.Tensor | None = None,
    ) -> dict:
        pb = pinball_loss(pred, target, self.quantiles, self.quantile_weights)
        mono = pred.new_tensor(0.0)
        if cell_ids is not None and cycles is not None and len(cell_ids) >= 2:
            mono = monotonic_penalty_per_cell(pred[:, 1], cell_ids, cycles)
        total = pb + self.lambda_mono * mono
        return {"loss": total, "pinball": pb.detach(), "mono": mono.detach()}


def picp(y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    inside = ((y_true >= lower) & (y_true <= upper)).float().mean()
    return float(inside.item())


def pinaw(y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    width = (upper - lower).mean()
    span = y_true.max() - y_true.min() + 1e-6
    return float((width / span).item())


def calibrate_interval_width(
    y_true: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor,
    upper: np.ndarray | torch.Tensor,
    target_picp: float = 0.90,
) -> float:
    """Find additive expansion ``c`` so PICP ≈ target (symmetric conformal)."""
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    # residual distance outside current interval
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    need = np.maximum(below, above)
    if float((need == 0).mean()) >= target_picp:
        return 0.0
    # quantile of needed expansion
    level = target_picp
    c = float(np.quantile(need, level))
    return max(c, 0.0)
