from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def make_grid(images: np.ndarray, nrow: int = 8) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    if images.ndim != 3:
        raise ValueError("Expected images with shape (N, H, W) or (N, H, W, 1).")

    n_images, height, width = images.shape
    ncol = nrow
    nrow_grid = int(math.ceil(n_images / ncol))

    grid = np.zeros((nrow_grid * height, ncol * width), dtype=images.dtype)
    for idx in range(n_images):
        r = idx // ncol
        c = idx % ncol
        grid[r * height : (r + 1) * height, c * width : (c + 1) * width] = images[idx]
    return grid


def save_image_grid(images: np.ndarray, path: Path, nrow: int = 8) -> None:
    grid = make_grid(images, nrow=nrow)
    grid = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(grid, mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
