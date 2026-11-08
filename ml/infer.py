"""Classify one frame with loaded weights.

The network assigns the whole frame to one class and does not predict object location.
So objects holds a single detection: the winning class, its confidence, and a box over the whole frame.
bbox is frame fractions from 0 to 1: x and y of the top-left corner, plus width and height.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from mlp import default_weights_path, load_checkpoint
from preprocess import flatten, normalize, rgb_from_bytes


class ClassScore(NamedTuple):
    name: str
    confidence: float


class BBox(NamedTuple):
    x: float
    y: float
    width: float
    height: float


class Detection(NamedTuple):
    name: str
    confidence: float
    bbox: BBox


# The whole frame area. This network cannot emit any other box.
FULL_FRAME_BBOX = BBox(0.0, 0.0, 1.0, 1.0)


class FramePrediction(NamedTuple):
    label: str
    confidence: float
    classes: tuple[ClassScore, ...]
    objects: tuple[Detection, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "classes": [
                {"name": item.name, "confidence": item.confidence}
                for item in self.classes
            ],
            "objects": [
                {
                    "class": item.name,
                    "confidence": item.confidence,
                    "bbox": {
                        "x": item.bbox.x,
                        "y": item.bbox.y,
                        "width": item.bbox.width,
                        "height": item.bbox.height,
                    },
                }
                for item in self.objects
            ],
        }


class Classifier:
    def __init__(self, weights_path: Path | None = None) -> None:
        path = default_weights_path() if weights_path is None else weights_path
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = load_checkpoint(path)
        self._model = checkpoint.model
        self._mean = checkpoint.mean
        self._std = checkpoint.std
        self._image_size = checkpoint.image_size
        self._class_names = checkpoint.class_names

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    def classify(self, image: bytes) -> FramePrediction:
        pixels = rgb_from_bytes(image, self._image_size)
        features = flatten(normalize(pixels, self._mean, self._std))
        probabilities = self._model.predict_proba(features)[0]
        scores = tuple(
            ClassScore(name, round(float(probability), 6))
            for name, probability in zip(self._class_names, probabilities, strict=True)
        )
        winner = scores[int(np.argmax(probabilities))]
        detection = Detection(
            name=winner.name, confidence=winner.confidence, bbox=FULL_FRAME_BBOX
        )
        return FramePrediction(
            label=winner.name,
            confidence=winner.confidence,
            classes=scores,
            objects=(detection,),
        )
