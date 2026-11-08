"""Обучение трёхслойной сети на Pascal VOC (dictvv/voc-yolo)."""

import argparse
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlp import Mlp, adam_step, cross_entropy, default_weights_path, new_adam_state, save_checkpoint, verify_gradients
from preprocess import channel_stats, flatten, load_rgb, normalize

REPO_ID = "dictvv/voc-yolo"
DATASET_DIR = Path(__file__).resolve().parent / "data" / "voc-yolo"
ARCHIVE_NAME = "voc.tar"

CLASS_NAMES = ("человек", "транспорт", "животное")
# Индексы YOLO из VOC: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, ...
VOC_TO_SUPER = {
    0: 1,
    1: 1,
    2: 2,
    3: 1,
    5: 1,
    6: 1,
    7: 2,
    9: 2,
    11: 2,
    12: 2,
    13: 1,
    14: 0,
    16: 2,
    18: 1,
}
TRAIN_SPLITS = ("train2007", "val2007", "train2012", "val2012")
VAL_SPLITS = ("test2007",)

IMAGE_SIZE = 32
HIDDEN_1 = 128
HIDDEN_2 = 32
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 8
SEED = 42


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def find_yolo_dirs(root: Path) -> tuple[Path, Path] | None:
    for images_dir in root.rglob("images"):
        if not images_dir.is_dir() or ".cache" in images_dir.parts:
            continue
        labels_dir = images_dir.parent / "labels"
        train_dir = images_dir / TRAIN_SPLITS[0]
        if train_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
    return None


def ensure_dataset(directory: Path) -> Path:
    if find_yolo_dirs(directory) is not None:
        print(f"датасет уже есть: {directory}")
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / ARCHIVE_NAME
    if not archive.is_file():
        print(f"скачиваю {REPO_ID}...")
        from huggingface_hub import hf_hub_download

        archive = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=ARCHIVE_NAME,
                repo_type="dataset",
                local_dir=directory,
            )
        )

    print(f"распаковываю {archive.name}...")
    with tarfile.open(archive, "r:") as tar:
        members = [member for member in tar.getmembers() if "VOCdevkit" not in Path(member.name).parts]
        tar.extractall(path=directory, members=members, filter="data")
    if find_yolo_dirs(directory) is None:
        raise FileNotFoundError(f"после распаковки не нашлись images/{TRAIN_SPLITS[0]} и labels")
    archive.unlink(missing_ok=True)
    return directory


def dominant_label(label_path: Path) -> int | None:
    areas = np.zeros(3, dtype=np.float64)
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        superclass = VOC_TO_SUPER.get(int(float(parts[0])))
        if superclass is None:
            continue
        box_area = float(parts[3]) * float(parts[4])
        if box_area <= 0.0:
            continue
        areas[superclass] += box_area
    if float(areas.sum()) <= 0.0:
        return None
    return int(np.argmax(areas))


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_split(images_dir: Path, labels_dir: Path, split: str) -> list[Sample]:
    image_dir = images_dir / split
    label_dir = labels_dir / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"нет сплита {split}")

    samples: list[Sample] = []
    missing = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        label = dominant_label(label_path)
        if label is None:
            continue
        image_path = find_image(image_dir, label_path.stem)
        if image_path is None:
            missing += 1
            continue
        samples.append(Sample(path=image_path, label=label))
    if missing:
        print(f"{split}: пропущено кадров без файла: {missing}")
    return samples


def load_samples(root: Path) -> tuple[list[Sample], list[Sample]]:
    found = find_yolo_dirs(root)
    if found is None:
        raise FileNotFoundError(f"нет YOLO-разметки VOC в {root}")
    images_dir, labels_dir = found

    train_samples: list[Sample] = []
    for split in TRAIN_SPLITS:
        train_samples.extend(load_split(images_dir, labels_dir, split))
    val_samples: list[Sample] = []
    for split in VAL_SPLITS:
        val_samples.extend(load_split(images_dir, labels_dir, split))

    if len(train_samples) < 20 or len(val_samples) < 20:
        raise RuntimeError(f"слишком мало кадров: train {len(train_samples)}, val {len(val_samples)}")
    return train_samples, val_samples


def class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    if np.any(counts == 0):
        missing = [CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
        raise RuntimeError(f"в обучении нет примеров класса: {', '.join(missing)}")
    return counts.sum() / (len(CLASS_NAMES) * counts)


def load_images(samples: list[Sample], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    for index, sample in enumerate(samples, start=1):
        images.append(load_rgb(sample.path, image_size))
        if index % 2000 == 0 or index == len(samples):
            print(f"загружено кадров: {index}/{len(samples)}")
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    return np.stack(images), labels


def format_counts(counts: np.ndarray) -> str:
    parts = [f"{name} {int(count)}" for name, count in zip(CLASS_NAMES, counts)]
    return ", ".join(parts)


def run_epoch(
    model: Mlp,
    inputs: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
    adam_state: dict[str, dict[str, np.ndarray | float]],
    rng: np.random.Generator,
    step: int,
) -> tuple[float, int]:
    order = rng.permutation(inputs.shape[0])
    loss_sum = 0.0
    seen = 0
    for start in range(0, order.size, BATCH_SIZE):
        batch_index = order[start : start + BATCH_SIZE]
        batch_inputs = inputs[batch_index]
        batch_labels = labels[batch_index]
        probabilities, cache = model.forward(batch_inputs)
        batch_loss = cross_entropy(probabilities, batch_labels, class_weights)
        grads = model.backward(cache, batch_labels, class_weights)
        step += 1
        adam_step(model, grads, adam_state, step, LEARNING_RATE, WEIGHT_DECAY)
        count = int(batch_labels.shape[0])
        loss_sum += batch_loss * count
        seen += count
    return loss_sum / seen, step


def evaluate(model: Mlp, inputs: np.ndarray, labels: np.ndarray, class_weights: np.ndarray) -> tuple[float, float, np.ndarray]:
    probabilities = model.predict_proba(inputs)
    predictions = probabilities.argmax(axis=1)
    loss = cross_entropy(probabilities, labels, class_weights)
    accuracy = float(np.mean(predictions == labels))
    return loss, accuracy, predictions


def per_class_recall(labels: np.ndarray, predictions: np.ndarray) -> list[float]:
    recalls: list[float] = []
    for class_id in range(len(CLASS_NAMES)):
        mask = labels == class_id
        if not np.any(mask):
            recalls.append(float("nan"))
            continue
        recalls.append(float(np.mean(predictions[mask] == class_id)))
    return recalls


def macro_recall(labels: np.ndarray, predictions: np.ndarray) -> float:
    valid = [value for value in per_class_recall(labels, predictions) if not np.isnan(value)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def train(epochs: int) -> Path:
    verify_gradients()
    print("градиенты совпали с численной проверкой")

    dataset_dir = ensure_dataset(DATASET_DIR)
    train_samples, val_samples = load_samples(dataset_dir)
    print("загружаю train...")
    train_images, train_labels = load_images(train_samples, IMAGE_SIZE)
    print("загружаю val...")
    val_images, val_labels = load_images(val_samples, IMAGE_SIZE)
    print(f"train: {len(train_samples)} ({format_counts(class_counts(train_labels))})")
    print(f"val: {len(val_samples)} ({format_counts(class_counts(val_labels))})")

    rng = np.random.default_rng(SEED)
    mean, std = channel_stats(train_images)
    train_inputs = flatten(normalize(train_images, mean, std))
    val_inputs = flatten(normalize(val_images, mean, std))
    weights = class_weights_from_counts(class_counts(train_labels))

    model = Mlp(
        input_size=train_inputs.shape[1],
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        output_size=len(CLASS_NAMES),
        rng=rng,
    )
    print(
        f"слои: {model.w1.shape[0]} -> {model.w1.shape[1]} -> {model.w2.shape[1]} -> {model.w3.shape[1]}, "
        f"параметров {model.parameter_count()}"
    )
    print(f"train {train_labels.size}, val {val_labels.size}")

    adam_state = new_adam_state(model)
    best_score = -1.0
    best_loss = float("inf")
    best_parameters = model.clone_parameters()
    best_predictions = np.zeros_like(val_labels)
    best_accuracy = 0.0
    stale_epochs = 0
    step = 0

    for epoch in range(1, epochs + 1):
        train_loss, step = run_epoch(model, train_inputs, train_labels, weights, adam_state, rng, step)
        val_loss, val_accuracy, val_predictions = evaluate(model, val_inputs, val_labels, weights)
        score = macro_recall(val_labels, val_predictions)
        print(
            f"эпоха {epoch:02d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"accuracy {val_accuracy:.3f}  macro {score:.3f}"
        )
        improved = score > best_score or (score == best_score and val_loss < best_loss)
        if improved:
            best_score = score
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_parameters = model.clone_parameters()
            best_predictions = val_predictions.copy()
            stale_epochs = 0
            continue
        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"остановка на эпохе {epoch}: проверка не улучшалась {PATIENCE} эпох")
            break

    model.restore(best_parameters)
    weights_path = default_weights_path()
    save_checkpoint(weights_path, model, mean, std, IMAGE_SIZE, CLASS_NAMES)

    recalls = per_class_recall(val_labels, best_predictions)
    print(f"лучшая проверка: loss {best_loss:.4f}, accuracy {best_accuracy:.3f}, macro {best_score:.3f}")
    for name, recall in zip(CLASS_NAMES, recalls):
        print(f"  {name}: recall {recall:.3f}")
    print(f"веса: {weights_path}")
    return weights_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Обучить классификатор человек / транспорт / животное")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("epochs должен быть больше нуля")
    train(args.epochs)


if __name__ == "__main__":
    main()
