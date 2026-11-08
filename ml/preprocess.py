"""Подготовка изображения к входу сети."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def _rgb_array(image: Image.Image, image_size: int) -> np.ndarray:
    converted = image.convert("RGB")
    resized = converted.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float64)
    return pixels / 255.0


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        return _rgb_array(image, image_size)


def rgb_from_bytes(data: bytes, image_size: int) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return _rgb_array(image, image_size)


def channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    return mean, np.maximum(std, 1e-6)


def normalize(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (images - mean) / std


def flatten(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        return images.reshape(1, -1)
    return images.reshape(images.shape[0], -1)
