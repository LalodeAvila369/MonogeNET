"""Tests for metrics, config, seeding and the training loop."""

import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from monogenet.train import TrainConfig, accuracy, auc_score, evaluate, fit, set_seed


def test_auc_known_values() -> None:
    y = torch.tensor([0, 0, 1, 1])
    assert auc_score(torch.tensor([0.1, 0.4, 0.35, 0.8]), y) == pytest.approx(0.75)  # classic example
    assert auc_score(torch.tensor([0.1, 0.2, 0.3, 0.4]), y) == 1.0
    assert auc_score(torch.tensor([0.4, 0.3, 0.2, 0.1]), y) == 0.0
    assert auc_score(torch.zeros(4), y) == 0.5  # all ties
    assert np.isnan(auc_score(torch.rand(3), torch.zeros(3)))  # one class only


def test_auc_matches_brute_force() -> None:
    g = torch.Generator().manual_seed(0)
    s, y = torch.randn(40, generator=g).round(decimals=1), torch.randint(0, 2, (40,), generator=g)
    p, n = s[y == 1], s[y == 0]
    brute = ((p[:, None] > n[None]).double() + 0.5 * (p[:, None] == n[None]).double()).mean()
    assert auc_score(s, y) == pytest.approx(float(brute))


def test_auc_two_column_and_multiclass() -> None:
    y = torch.tensor([0, 1, 1, 0])
    z = torch.tensor([[2.0, 0], [0, 1], [0, 3], [1, 0]])
    assert auc_score(z, y) == 1.0
    onehot = torch.eye(4)[torch.arange(8) % 4] * 5
    assert auc_score(onehot, torch.arange(8) % 4) == 1.0
    assert auc_score(torch.zeros(8, 4), torch.arange(8) % 4) == pytest.approx(0.5)


def test_accuracy() -> None:
    assert accuracy(torch.tensor([[1.0, 0], [0, 1], [3, 0]]), torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)
    assert accuracy(torch.tensor([1.0, -1.0]), torch.tensor([1, 0])) == 1.0


def test_set_seed_reproducible() -> None:
    set_seed(5)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(5)
    assert a == (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_config_roundtrip(tmp_path) -> None:
    cfg = TrainConfig(model="resnet18", lr=3e-4, extra={"scales": 3})
    path = str(tmp_path / "c.json")
    cfg.save(path)
    assert TrainConfig.load(path) == cfg
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"lr": 1, "nope": 2}))
    with pytest.raises(ValueError):
        TrainConfig.load(str(bad))


def _toy(n: int = 64, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = torch.randn(n, 1, 8, 8, generator=g) * 0.3 + (y.float() * 2 - 1).view(-1, 1, 1, 1)
    return DataLoader(TensorDataset(x, y.view(-1, 1)), batch_size=16)


def _net() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 2))


def test_fit_learns_and_is_deterministic() -> None:
    cfg = TrainConfig(epochs=8, lr=0.05, seed=3)
    runs = []
    for _ in range(2):
        set_seed(cfg.seed)
        model = _net()
        hist = fit(model, _toy(), _toy(seed=1), cfg)
        runs.append((hist, model))
    h = runs[0][0]
    assert len(h) == 8 and h[-1]["train_loss"] < h[0]["train_loss"]
    assert h[-1]["val_acc"] > 0.95 and h[-1]["val_auc"] > 0.95
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1].state_dict().values(), runs[1][1].state_dict().values()))


def test_fit_restores_best_and_calls_callback() -> None:
    seen = []
    model = _net()
    fit(model, _toy(), _toy(seed=1), TrainConfig(epochs=3, lr=0.05), on_epoch=seen.append)
    assert [r["epoch"] for r in seen] == [0, 1, 2]
    m = evaluate(model, _toy(seed=1))
    assert set(m) == {"loss", "acc", "auc"} and m["auc"] >= max(r["val_auc"] for r in seen) - 1e-9


def test_fit_without_validation() -> None:
    hist = fit(_net(), _toy(), None, TrainConfig(epochs=2))
    assert "val_auc" not in hist[0] and np.isfinite(hist[0]["train_loss"])
