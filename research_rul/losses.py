"""Pinball loss and monotonic RUL penalty."""

from __future__ import annotations

import torch
import torch.nn as nn


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    pred: (B, Q), target: (B,), quantiles: (Q,)
    """
    target = target.unsqueeze(-1)
    errors = target - pred
    q = quantiles.view(1, -1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
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
    """
    Penalize RUL increases along batch time order.
    Caller must pass predictions sorted by cycle within each cell batch,
    or use per-sequence penalty in the model forward.
    """
    if pred_median.numel() < 2:
        return pred_median.new_tensor(0.0)
    diff = pred_median[1:] - pred_median[:-1]
    return torch.relu(diff).pow(2).mean()


class QuantileRULLoss(nn.Module):
    def __init__(self, quantiles=(0.05, 0.5, 0.95), lambda_mono: float = 0.1):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.lambda_mono = lambda_mono

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cell_ids: list[str] | None = None,
        cycles: torch.Tensor | None = None,
    ) -> dict:
        pb = pinball_loss(pred, target, self.quantiles)
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
