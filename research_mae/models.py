"""MS-CNN masked autoencoder + channel-attention multimodal fusion."""

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


class MSConvBlock(nn.Module):
    """Parallel conv branches (kernels 3/5/7) for multi-scale temporal patterns."""

    def __init__(self, in_ch: int, out_ch: int, kernels: tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        branch_ch = max(out_ch // len(kernels), 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_ch, branch_ch, k, padding=k // 2),
                    nn.BatchNorm1d(branch_ch),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.merge = nn.Conv1d(branch_ch * len(kernels), out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([b(x) for b in self.branches], dim=1))


class MSCNNMaskedAE(nn.Module):
    """Multi-scale 1D-CNN masked autoencoder for relaxation ΔV sequences."""

    def __init__(self, seq_len: int = 32, latent_dim: int = 32, mask_ratio: float = 0.3):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio

        self.encoder = nn.Sequential(
            MSConvBlock(1, 32),
            MSConvBlock(32, 64),
            MSConvBlock(64, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.to_latent = nn.Linear(128, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * seq_len)

        self.decoder = nn.Sequential(
            ConvBlock(128, 64, 5),
            ConvBlock(64, 32, 5),
            ConvBlock(32, 16, 3),
            nn.Conv1d(16, 1, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(z.size(0), 128, self.seq_len)
        return self.decoder(h)

    def random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, length = x.shape
        n_mask = max(1, int(length * self.mask_ratio))
        mask = torch.ones(b, length, device=x.device)
        for i in range(b):
            idx = torch.randperm(length, device=x.device)[:n_mask]
            mask[i, idx] = 0.0
        mask = mask.unsqueeze(1)
        return x * mask, mask

    def forward(self, x: torch.Tensor) -> dict:
        masked, mask = self.random_mask(x)
        z = self.encode(masked)
        recon = self.decode(z)
        return {"recon": recon, "mask": mask, "latent": z}


# Backward-compatible alias
TemporalMaskedAE = MSCNNMaskedAE


class ChannelAttentionFusion(nn.Module):
    """
    Softmax channel attention over relaxation latent vs CC macro features.
    Weights sum to 1 and adapt to the current aging stage.
    """

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
    """Legacy sigmoid gates (kept for loading old checkpoints)."""

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

    def __init__(self, latent_dim: int = 32, cc_feat_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        in_dim = latent_dim * 2 + cc_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(
        self,
        fused: torch.Tensor,
        relax_latent: torch.Tensor,
        cc_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([fused, relax_latent, cc_feat], dim=-1)
        return self.net(x).squeeze(-1)


def train_mae_epoch(model, loader, optimizer, device, max_grad_norm: float = 1.0) -> dict:
    model.train()
    totals = {"mse": 0.0, "smooth": 0.0, "loss": 0.0}
    n = 0
    for batch in loader:
        x = batch[0].to(device)
        out = model(x)
        mask = out["mask"]
        recon = out["recon"]
        mse = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        smooth = torch.mean((recon[:, :, 1:] - recon[:, :, :-1]) ** 2)
        loss = mse + 0.05 * smooth
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        totals["mse"] += float(mse.item()) * x.size(0)
        totals["smooth"] += float(smooth.item()) * x.size(0)
        totals["loss"] += float(loss.item()) * x.size(0)
        n += x.size(0)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def infer_latent(model: MSCNNMaskedAE, x: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    return model.encode(x.to(device)).cpu()
