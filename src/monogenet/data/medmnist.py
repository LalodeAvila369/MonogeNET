"""MedMNIST+ loaders (BreastMNIST, OCTMNIST) at sizes 28, 64, 128 and 224.

Images are returned as ``float32`` tensors ``(1, H, W)`` scaled to ``[0, 1]`` and
labels as ``int64`` scalars. The uint8 arrays stay in memory (or memory-mapped), and
each sample is converted on access, so OCTMNIST-224 (about 5 GB) is not duplicated
in float. Data are never downloaded unless ``download=True``.

Please cite MedMNIST v1 (Yang et al., ISBI 2021) and v2 (Yang et al., Scientific
Data 2023) when using these datasets.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["DATASETS", "MedMNISTDataset", "load_split", "dataloaders"]

DATASETS = {"breastmnist": "BreastMNIST", "octmnist": "OCTMNIST"}
SIZES = (28, 64, 128, 224)
SPLITS = ("train", "val", "test")


def default_root() -> str:
    """Data directory: ``$MONOGENET_DATA`` or ``~/.medmnist`` (the medmnist default)."""
    return os.environ.get("MONOGENET_DATA", os.path.join(os.path.expanduser("~"), ".medmnist"))


class MedMNISTDataset(Dataset):
    """A split of a MedMNIST+ dataset as ``(image, label)`` tensors.

    ``image`` has shape ``(1, size, size)`` in ``[0, 1]`` (grayscale), ``label`` is a
    scalar ``int64`` class index (binary for BreastMNIST, 4 classes for OCTMNIST).
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        size: int = 128,
        root: str | None = None,
        download: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mmap: bool = False,
    ) -> None:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}, got {size}")
        import medmnist  # optional dependency: pip install monogenet[data]

        root = root or default_root()
        os.makedirs(root, exist_ok=True)
        cls = getattr(medmnist, DATASETS[name])
        ds: Any = cls(split=split, root=root, size=size, download=download, mmap_mode="r" if mmap else None)
        self.name, self.split, self.size = name, split, size
        self.info = ds.info
        self.n_classes = len(ds.info["label"])
        self.transform = transform
        self._images: np.ndarray = ds.imgs  # (N, H, W) uint8
        self._labels = torch.from_numpy(np.asarray(ds.labels).reshape(len(ds.imgs), -1)[:, 0].astype(np.int64))

    @property
    def labels(self) -> torch.Tensor:
        """All labels, shape ``(N,)``."""
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(np.array(self._images[index], dtype=np.float32) / 255.0)
        img = img.unsqueeze(0) if img.ndim == 2 else img.permute(2, 0, 1)[:1]
        if self.transform is not None:
            img = self.transform(img)
        return img, self._labels[index]


def load_split(name: str, split: str = "train", size: int = 128, **kwargs: Any) -> MedMNISTDataset:
    """Shorthand for :class:`MedMNISTDataset`."""
    return MedMNISTDataset(name, split, size, **kwargs)


def dataloaders(
    name: str,
    size: int = 128,
    batch_size: int = 64,
    *,
    seed: int = 0,
    num_workers: int = 0,
    **kwargs: Any,
) -> dict[str, DataLoader]:
    """Train (shuffled, seeded), validation and test loaders."""
    out: dict[str, DataLoader] = {}
    for split in SPLITS:
        ds = MedMNISTDataset(name, split, size, **kwargs)
        gen = torch.Generator().manual_seed(seed)
        out[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=split == "train", generator=gen if split == "train" else None,
            num_workers=num_workers,
        )
    return out
