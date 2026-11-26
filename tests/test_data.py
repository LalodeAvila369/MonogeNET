"""Tests for the MedMNIST loaders, using small fake .npz files (no download)."""

import numpy as np
import pytest
import torch

pytest.importorskip("medmnist")

from monogenet.data import MedMNISTDataset, dataloaders, load_split  # noqa: E402


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    for flag, classes, size in (("breastmnist", 2, 128), ("octmnist", 4, 64)):
        arrays = {}
        for split, n in (("train", 12), ("val", 4), ("test", 5)):
            arrays[f"{split}_images"] = rng.integers(0, 256, (n, size, size), dtype=np.uint8)
            arrays[f"{split}_labels"] = (np.arange(n) % classes).reshape(n, 1).astype(np.uint8)
        np.savez(tmp_path / f"{flag}_{size}.npz", **arrays)
    return str(tmp_path)


def test_item_format(root) -> None:
    ds = MedMNISTDataset("breastmnist", "train", 128, root=root)
    x, y = ds[3]
    assert len(ds) == 12 and ds.n_classes == 2
    assert x.shape == (1, 128, 128) and x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1
    assert y.dtype == torch.int64 and y.shape == () and int(y) == 1
    assert ds.labels.shape == (12,)


def test_values_scaled_from_uint8(root) -> None:
    ds = load_split("octmnist", "val", 64, root=root)
    raw = np.load(f"{root}/octmnist_64.npz")["val_images"]
    assert torch.allclose(ds[2][0][0], torch.from_numpy(raw[2] / 255.0).float())
    assert ds.n_classes == 4 and len(ds) == 4


def test_mmap_and_transform(root) -> None:
    ds = MedMNISTDataset("breastmnist", "test", 128, root=root, mmap=True, transform=lambda t: t * 2)
    assert len(ds) == 5 and ds[0][0].max() <= 2.0 and ds[0][0].max() > 1.0


def test_dataloaders_batches_and_seeded_shuffle(root) -> None:
    a = dataloaders("breastmnist", 128, batch_size=5, seed=1, root=root)
    b = dataloaders("breastmnist", 128, batch_size=5, seed=1, root=root)
    xa, ya = next(iter(a["train"]))
    xb, yb = next(iter(b["train"]))
    assert xa.shape == (5, 1, 128, 128) and torch.equal(ya, yb) and torch.equal(xa, xb)
    assert set(a) == {"train", "val", "test"} and sum(len(y) for _, y in a["test"]) == 5


def test_validation(root) -> None:
    with pytest.raises(ValueError):
        MedMNISTDataset("pathmnist", root=root)
    with pytest.raises(ValueError):
        MedMNISTDataset("breastmnist", "dev", root=root)
    with pytest.raises(ValueError):
        MedMNISTDataset("breastmnist", size=100, root=root)
    with pytest.raises(RuntimeError):
        MedMNISTDataset("breastmnist", "train", 224, root=root)  # file missing, no download
