"""Check a trained model on a single image."""

import argparse
import sys
from pathlib import Path

from PIL import UnidentifiedImageError

from infer import Classifier
from mlp import default_weights_path


def predict(image_path: Path, weights_path: Path) -> int:
    if not image_path.is_file():
        print(f"file not found: {image_path}")
        return 1
    try:
        classifier = Classifier(weights_path)
    except FileNotFoundError:
        print("weights not found. Train first: python ml/train.py")
        return 1
    try:
        prediction = classifier.classify(image_path.read_bytes())
    except (UnidentifiedImageError, OSError):
        print(f"failed to read image: {image_path}")
        return 1

    print(image_path)
    for item in prediction.classes:
        print(f"{item.name:<12} {item.confidence:.3f}")
    print(f"answer: {prediction.label}")
    for item in prediction.objects:
        box = item.bbox
        print(
            f"object: {item.name} {item.confidence:.3f}  "
            f"bbox x={box.x:.3f} y={box.y:.3f} w={box.width:.3f} h={box.height:.3f}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an image with the trained model")
    parser.add_argument("image", type=Path, help="path to a jpg, png, or other image file")
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(predict(args.image, args.weights))


if __name__ == "__main__":
    main()
