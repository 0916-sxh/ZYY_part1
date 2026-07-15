"""Export per-cycle fused features to static .npy files for downstream RUL modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_mae.evaluate import normalize_latent, prepare_cc_tensor
from research_mae.features import build_cc_features, normalize_cc_features
from research_mae.models import GatedChannelFusion, MSCNNMaskedAE, infer_latent

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"


@torch.no_grad()
def compute_fused_features(
    data: dict,
    mae: MSCNNMaskedAE,
    fusion: GatedChannelFusion,
    stats: dict,
    device: str = "cpu",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Encode relaxation sequences and fuse with CC macro features."""
    x = torch.from_numpy(data["delta_v"]).unsqueeze(1)
    n = len(x)
    latent_parts, fused_parts, weight_parts = [], [], []

    cc_tensor = torch.from_numpy(
        prepare_cc_tensor(data["cc_time_s"], data["cell_id"], data["cycle"], stats)
    ).float()

    mae.eval()
    fusion.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_b = x[start:end]
        z_raw = infer_latent(mae, x_b, device).numpy()
        z_norm = normalize_latent(z_raw, stats)
        z_t = torch.from_numpy(z_norm).float().to(device)
        cc_b = cc_tensor[start:end].to(device)

        fused_b, weights_b = fusion(z_t, cc_b)
        latent_parts.append(z_norm.astype(np.float32))
        fused_parts.append(fused_b.cpu().numpy().astype(np.float32))
        weight_parts.append(weights_b.cpu().numpy().astype(np.float32))

    return {
        "latent": np.concatenate(latent_parts, axis=0),
        "fused": np.concatenate(fused_parts, axis=0),
        "attention": np.concatenate(weight_parts, axis=0),
    }


def export_dataset_features(
    data: dict,
    dataset_id: int,
    mae: MSCNNMaskedAE,
    fusion: GatedChannelFusion,
    stats: dict,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Save fused/latent features and metadata for one dataset.

    Outputs (per dataset):
      - dataset_{id}_fused.npy      (N, latent_dim)  — downstream RUL input
      - dataset_{id}_latent.npy     (N, latent_dim)  — relaxation-only latent
      - dataset_{id}_meta.npz       cell_id, cycle, capacity, cc_time_s, attention
      - dataset_{id}_manifest.json schema + shapes
    """
    out_dir = out_dir or FEATURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = compute_fused_features(data, mae, fusion, stats, device=device)
    prefix = f"dataset_{dataset_id}"

    fused_path = out_dir / f"{prefix}_fused.npy"
    latent_path = out_dir / f"{prefix}_latent.npy"
    meta_path = out_dir / f"{prefix}_meta.npz"

    np.save(fused_path, feats["fused"])
    np.save(latent_path, feats["latent"])
    np.savez_compressed(
        meta_path,
        dataset=np.full(len(data["cycle"]), dataset_id, dtype=np.int8),
        cell_id=data["cell_id"],
        cycle=data["cycle"].astype(np.int32),
        capacity=data["capacity"].astype(np.float32),
        cc_time_s=data["cc_time_s"].astype(np.float32),
        attention_relax=feats["attention"][:, 0].astype(np.float32),
        attention_cc=feats["attention"][:, 1].astype(np.float32),
    )

    manifest = {
        "dataset_id": dataset_id,
        "n_samples": int(len(data["cycle"])),
        "latent_dim": int(feats["fused"].shape[1]),
        "files": {
            "fused": fused_path.name,
            "latent": latent_path.name,
            "meta": meta_path.name,
        },
        "shapes": {
            "fused": list(feats["fused"].shape),
            "latent": list(feats["latent"].shape),
        },
        "description": "Research content 1: Dilated MS-CNN latent + channel-attention fused features",
    }
    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "fused": fused_path,
        "latent": latent_path,
        "meta": meta_path,
        "manifest": manifest_path,
    }


def export_all(
    datasets: dict[int, dict],
    mae_short: MSCNNMaskedAE,
    mae_long: MSCNNMaskedAE,
    fusion_by_ds: dict[int, tuple[GatedChannelFusion, dict]],
    device: str = "cpu",
    out_dir: Path | None = None,
) -> dict[int, dict[str, Path]]:
    """Export features for datasets 1–3."""
    mae_map = {1: mae_short, 2: mae_short, 3: mae_long}
    paths = {}
    for ds_id in (1, 2, 3):
        if ds_id not in datasets or ds_id not in fusion_by_ds:
            continue
        fusion, stats = fusion_by_ds[ds_id]
        paths[ds_id] = export_dataset_features(
            datasets[ds_id],
            ds_id,
            mae_map[ds_id],
            fusion,
            stats,
            device=device,
            out_dir=out_dir,
        )
        print(f"  Dataset {ds_id}: {paths[ds_id]['fused'].name}  ({len(datasets[ds_id]['cycle'])} samples)")
    return paths


def load_fused_features(dataset_id: int, feature_dir: Path | None = None) -> dict:
    """Load exported features for downstream research (content 2/3)."""
    feature_dir = feature_dir or FEATURE_DIR
    prefix = f"dataset_{dataset_id}"
    fused = np.load(feature_dir / f"{prefix}_fused.npy")
    latent = np.load(feature_dir / f"{prefix}_latent.npy")
    meta = np.load(feature_dir / f"{prefix}_meta.npz", allow_pickle=True)
    return {
        "fused": fused,
        "latent": latent,
        "cell_id": meta["cell_id"],
        "cycle": meta["cycle"],
        "capacity": meta["capacity"],
        "cc_time_s": meta["cc_time_s"],
        "attention": np.stack([meta["attention_relax"], meta["attention_cc"]], axis=1),
    }
