r"""Training loop, metrics (ACC, AUC), JSON configs and seeding.

AUC is the area under the ROC curve, computed from the Mann-Whitney statistic
:math:`\mathrm{AUC} = \frac{1}{PN}\sum_{i\in P}\sum_{j\in N}
[\mathbb 1(s_i > s_j) + \tfrac12 \mathbb 1(s_i = s_j)]`. For more than two classes
it is the macro average of one-vs-rest AUCs (the MedMNIST convention).
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = ["TrainConfig", "set_seed", "accuracy", "auc_score", "evaluate", "fit"]


@dataclass
class TrainConfig:
    """Hyper-parameters of a run. Save and load as JSON with ``save`` / ``load``."""

    dataset: str = "breastmnist"
    size: int = 128
    model: str = "small_cnn"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy. ``logits`` is ``(N, K)`` (or ``(N,)`` for a binary logit)."""
    pred = (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)
    return float((pred == labels.long()).float().mean())


def _binary_auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    neg_sorted = neg.sort().values
    below = torch.searchsorted(neg_sorted, pos, right=False)
    upto = torch.searchsorted(neg_sorted, pos, right=True)
    return float((below.double() + 0.5 * (upto - below).double()).sum() / (len(pos) * len(neg)))


def auc_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC: binary, or macro one-vs-rest for ``(N, K)`` scores with ``K > 2``.

    Classes absent from ``labels`` are skipped; returns NaN if none is usable.
    """
    labels = labels.long()
    if logits.ndim == 1 or logits.shape[1] == 2:
        score = logits if logits.ndim == 1 else logits[:, 1] - logits[:, 0]
        return _binary_auc(score.detach(), labels == 1)
    probs = logits.detach().softmax(dim=1)
    aucs = [_binary_auc(probs[:, k], labels == k) for k in range(probs.shape[1])]
    aucs = [a for a in aucs if a == a]
    return sum(aucs) / len(aucs) if aucs else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: str = "cpu") -> dict[str, float]:
    """Loss, accuracy and AUC over a loader (model put in eval mode)."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y.reshape(-1).cpu())
    z, y = torch.cat(logits), torch.cat(labels)
    loss = nn.functional.cross_entropy(z, y.long())
    return {"loss": float(loss), "acc": accuracy(z, y), "auc": auc_score(z, y)}


def fit(
    model: nn.Module,
    train: Iterable,
    val: Iterable | None = None,
    config: TrainConfig | None = None,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Train with Adam, a cosine schedule and cross-entropy; one record per epoch.

    With ``val``, the best-validation-AUC weights (ties: lowest val loss) are
    restored at the end. Records hold ``epoch``, ``train_loss`` and, with ``val``,
    ``val_loss``, ``val_acc`` and ``val_auc``.
    """
    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    history: list[dict[str, Any]] = []
    best: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for x, y in train:
            x, y = x.to(cfg.device), y.reshape(-1).long().to(cfg.device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        sched.step()
        rec: dict[str, Any] = {"epoch": epoch, "train_loss": total / max(count, 1)}
        if val is not None:
            m = evaluate(model, val, cfg.device)
            rec.update({f"val_{k}": v for k, v in m.items()})
            key = (m["auc"] if m["auc"] == m["auc"] else -1.0, -m["loss"])
            if best is None or key > best:
                best, best_state = key, {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        if on_epoch is not None:
            on_epoch(rec)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
