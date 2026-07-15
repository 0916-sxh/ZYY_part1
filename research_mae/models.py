"""Hybrid Dilated MS-CNN masked autoencoder + gated multimodal fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedMSConvBlock(nn.Module):
    """Hybrid multi-scale block: local kernels + dilated kernels, with residual.

    Branches cover both short-range polarization shapes (kernels 3/5/7) and
    longer contexts via dilated kernel-3 (dilation 2/4), then 1×1 merge.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels: tuple[int, ...] = (3, 5, 7),
        dilations: tuple[int, ...] = (2, 4),
    ):
        super().__init__()
        specs: list[tuple[int, int]] = [(k, 1) for k in kernels] + [(3, d) for d in dilations]
        branch_ch = max(out_ch // len(specs), 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_ch, branch_ch, k, padding=d * (k // 2), dilation=d),
                    nn.BatchNorm1d(branch_ch),
                    nn.GELU(),
                )
                for k, d in specs
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv1d(branch_ch * len(specs), out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.merge(torch.cat([b(x) for b in self.branches], dim=1))
        return y + self.skip(x)


# Backward-compatible name
MSConvBlock = DilatedMSConvBlock


class MSCNNMaskedAE(nn.Module):
    """Dilated / hybrid multi-scale 1D-CNN masked autoencoder for ΔV sequences."""

    def __init__(
        self,
        seq_len: int = 32,
        latent_dim: int = 32,
        mask_ratio: float = 0.3,
        dilations: tuple[int, ...] = (2, 4),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        self.dilations = dilations

        self.encoder_stem = nn.Sequential(
            DilatedMSConvBlock(1, 32, dilations=dilations),
            DilatedMSConvBlock(32, 64, dilations=dilations),
            DilatedMSConvBlock(64, 128, dilations=dilations),
        )
        # Dual pooling preserves both average trend and peak polarization
        self.to_latent = nn.Linear(256, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * seq_len)

        self.decoder = nn.Sequential(
            ConvBlock(128, 64, 5),
            ConvBlock(64, 32, 5),
            ConvBlock(32, 16, 3),
            nn.Conv1d(16, 1, 3, padding=1),
        )
        # Soft aging axis: latent→SOH head pulls manifold toward monotonic aging
        self.aging_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),  # keep aging score in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_stem(x)
        avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        mx = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return self.to_latent(torch.cat([avg, mx], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(z.size(0), 128, self.seq_len)
        return self.decoder(h)

    def predict_aging(self, z: torch.Tensor) -> torch.Tensor:
        return self.aging_head(z).squeeze(-1)

    def block_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask a contiguous block covering ``mask_ratio`` of the sequence length."""
        b, _, length = x.shape
        n_mask = max(1, int(round(length * self.mask_ratio)))
        n_mask = min(n_mask, length)
        mask = torch.ones(b, 1, length, device=x.device)
        max_start = length - n_mask
        for i in range(b):
            start = int(torch.randint(0, max_start + 1, (1,), device=x.device).item())
            mask[i, 0, start : start + n_mask] = 0.0
        return x * mask, mask

    def random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block_mask(x)

    def forward(self, x: torch.Tensor) -> dict:
        masked, mask = self.block_mask(x)
        z = self.encode(masked)
        recon = self.decode(z)
        aging = self.predict_aging(z)
        return {"recon": recon, "mask": mask, "latent": z, "aging": aging}


TemporalMaskedAE = MSCNNMaskedAE


class ChannelAttentionFusion(nn.Module):
    """Softmax channel attention over relaxation latent vs CC macro features."""

    def __init__(self, latent_dim: int = 32, cc_feat_dim: int = 2, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.cc_embed = nn.Sequential(
            nn.Linear(cc_feat_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.relax_proj = nn.Linear(latent_dim, latent_dim)
        self.attn = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2),
        )

    def forward(
        self, relax_latent: torch.Tensor, cc_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relax_feat = self.relax_proj(relax_latent)
        cc_emb = self.cc_embed(cc_feat)
        ctx = torch.cat([relax_feat, cc_emb], dim=-1)
        logits = self.attn(ctx) / self.temperature
        weights = F.softmax(logits, dim=-1)
        fused = weights[:, 0:1] * relax_feat + weights[:, 1:2] * cc_emb
        return fused, weights


class GatedChannelFusion(nn.Module):
    """Independent sigmoid gates (production fusion)."""

    def __init__(self, latent_dim: int = 32, cc_feat_dim: int = 2):
        super().__init__()
        self.cc_embed = nn.Sequential(
            nn.Linear(cc_feat_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.relax_proj = nn.Linear(latent_dim, latent_dim)
        self.gate_r = nn.Linear(latent_dim * 2, 1)
        self.gate_c = nn.Linear(latent_dim * 2, 1)

    def forward(
        self, relax_latent: torch.Tensor, cc_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relax_feat = self.relax_proj(relax_latent)
        cc_emb = self.cc_embed(cc_feat)
        ctx = torch.cat([relax_feat, cc_emb], dim=-1)
        g_r = torch.sigmoid(self.gate_r(ctx))
        g_c = torch.sigmoid(self.gate_c(ctx))
        fused = g_r * relax_feat + g_c * cc_emb
        denom = g_r + g_c + 1e-6
        weights = torch.cat([g_r / denom, g_c / denom], dim=-1)
        return fused, weights


class CapacityHead(nn.Module):
    """Predict capacity from fused features + skip connections."""

    def __init__(self, latent_dim: int = 32, cc_feat_dim: int = 2, dropout: float = 0.05):
        super().__init__()
        in_dim = latent_dim * 2 + cc_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self,
        fused: torch.Tensor,
        relax_latent: torch.Tensor,
        cc_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([fused, relax_latent, cc_feat], dim=-1)
        return self.net(x).squeeze(-1)


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Encourage predicted aging scores to preserve order of SOH/life targets."""
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    # (B,B) pairwise differences
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)
    dt = target.unsqueeze(1) - target.unsqueeze(0)
    # Only pairs with clear target order
    mask = (dt.abs() > 1e-3).float()
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    # Want sign(dp) == sign(dt): hinge on -dp*sign(dt)
    loss = F.relu(0.05 - dp * torch.sign(dt))
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def train_mae_epoch(
    model,
    loader,
    optimizer,
    device,
    max_grad_norm: float = 1.0,
    lambda_aging: float = 0.25,
    lambda_rank: float = 0.1,
) -> dict:
    model.train()
    totals = {"mse": 0.0, "smooth": 0.0, "aging": 0.0, "rank": 0.0, "loss": 0.0}
    n = 0
    for batch in loader:
        if len(batch) == 1:
            x = batch[0].to(device)
            aging_t = None
        else:
            x = batch[0].to(device)
            aging_t = batch[1].to(device)
        out = model(x)
        mask = out["mask"]
        recon = out["recon"]
        mse_m = ((recon - x) ** 2 * (1.0 - mask)).sum() / (1.0 - mask).sum().clamp(min=1.0)
        mse_v = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        mse = 0.85 * mse_m + 0.15 * mse_v
        smooth = torch.mean((recon[:, :, 1:] - recon[:, :, :-1]) ** 2)
        loss = mse + 0.05 * smooth
        aging_l = x.new_tensor(0.0)
        rank_l = x.new_tensor(0.0)
        if aging_t is not None and hasattr(model, "aging_head"):
            aging_l = F.smooth_l1_loss(out["aging"], aging_t)
            rank_l = pairwise_ranking_loss(out["aging"], aging_t)
            loss = loss + lambda_aging * aging_l + lambda_rank * rank_l
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        totals["mse"] += float(mse.item()) * x.size(0)
        totals["smooth"] += float(smooth.item()) * x.size(0)
        totals["aging"] += float(aging_l.item()) * x.size(0)
        totals["rank"] += float(rank_l.item()) * x.size(0)
        totals["loss"] += float(loss.item()) * x.size(0)
        n += x.size(0)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def infer_latent(model: MSCNNMaskedAE, x: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    return model.encode(x.to(device)).cpu()
