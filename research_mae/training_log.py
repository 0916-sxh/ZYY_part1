"""Training history logging and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class TrainHistory:
    name: str
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    extra: dict[str, list[float]] = field(default_factory=dict)

    def append(self, epoch: int, train: float, val: float, lr: float, **kwargs):
        self.epochs.append(epoch)
        self.train_loss.append(train)
        self.val_loss.append(val)
        self.lr.append(lr)
        for k, v in kwargs.items():
            self.extra.setdefault(k, []).append(float(v))

    def save(self, path: Path | None = None) -> Path:
        path = path or OUTPUT_DIR / f"history_{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> TrainHistory:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
