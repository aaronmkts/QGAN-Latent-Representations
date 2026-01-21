from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

MNIST_URLS = {
    "train_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
    ),
    "train_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
    ),
    "test_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
    ),
    "test_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    ),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dims = magic & 0xFF
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


def load_mnist(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    paths = {
        key: data_dir / Path(url).name for key, url in MNIST_URLS.items()
    }
    for key, url in MNIST_URLS.items():
        _download(url, paths[key])

    train_images = _read_idx(paths["train_images"])
    train_labels = _read_idx(paths["train_labels"])
    test_images = _read_idx(paths["test_images"])
    test_labels = _read_idx(paths["test_labels"])

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    return train_images, train_labels, test_images, test_labels


class MNISTDataModule:
    def __init__(self, data_dir: str | Path, num_workers: int = 0) -> None:
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def setup(self) -> None:
        (
            self.train_images,
            self.train_labels,
            self.test_images,
            self.test_labels,
        ) = load_mnist(self.data_dir)

    def train_batches(
        self, batch_size: int, shuffle: bool = True, seed: int | None = None
    ) -> Iterator[dict]:
        if self.train_images is None:
            raise RuntimeError("Call setup() before requesting batches.")
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.train_images))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "images": self.train_images[batch_idx],
                "labels": self.train_labels[batch_idx],
            }

    def test_batches(
        self, batch_size: int, shuffle: bool = False, seed: int | None = None
    ) -> Iterator[dict]:
        if self.test_images is None:
            raise RuntimeError("Call setup() before requesting batches.")
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.test_images))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "images": self.test_images[batch_idx],
                "labels": self.test_labels[batch_idx],
            }
