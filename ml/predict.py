"""Проверка обученной модели на одном изображении."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import UnidentifiedImageError

from mlp import default_weights_path, load_checkpoint
from preprocess import flatten, load_rgb, normalize


def predict(image_path: Path, weights_path: Path) -> int:
    if not image_path.is_file():
        print(f"файл не найден: {image_path}")
        return 1
    if not weights_path.is_file():
        print("веса не найдены. Сначала запусти: python ml/train.py")
        return 1

    model, mean, std, image_size, class_names = load_checkpoint(weights_path)
    try:
        image = load_rgb(image_path, image_size)
    except (UnidentifiedImageError, OSError):
        print(f"не удалось прочитать изображение: {image_path}")
        return 1
    features = flatten(normalize(image, mean, std))
    probabilities = model.predict_proba(features)[0]

    print(image_path)
    for name, probability in zip(class_names, probabilities):
        print(f"{name:<12} {probability:.3f}")
    winner = class_names[int(np.argmax(probabilities))]
    print(f"ответ: {winner}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Классифицировать изображение обученной моделью")
    parser.add_argument("image", type=Path, help="путь к jpg, png или другому файлу изображения")
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(predict(args.image, args.weights))


if __name__ == "__main__":
    main()
