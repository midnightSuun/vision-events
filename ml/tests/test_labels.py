"""Superclass mapping for Pascal VOC labels."""

from pathlib import Path

from train import CLASS_NAMES, VOC_TO_SUPER, dominant_label


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_dominant_label_uses_superclass_by_area(tmp_path: Path) -> None:
    person = _write(tmp_path / "person.txt", "14 0.5 0.5 0.4 0.5\n4 0.5 0.5 0.9 0.9\n")
    vehicle = _write(
        tmp_path / "vehicle.txt", "14 0.5 0.5 0.2 0.2\n6 0.5 0.5 0.8 0.8\n"
    )
    animal = _write(tmp_path / "animal.txt", "11 0.5 0.5 0.3 0.3\n")
    clutter = _write(tmp_path / "clutter.txt", "8 0.5 0.5 0.5 0.5\n")

    assert dominant_label(person) == CLASS_NAMES.index("person")
    assert dominant_label(vehicle) == CLASS_NAMES.index("vehicle")
    assert dominant_label(animal) == CLASS_NAMES.index("animal")
    assert dominant_label(clutter) is None


def test_voc_indexes_stay_on_the_same_superclass() -> None:
    assert VOC_TO_SUPER == {
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
