"""Gradient check and checkpoint loading for the three-layer network."""

from pathlib import Path

import numpy as np

from mlp import Mlp, load_checkpoint, save_checkpoint


def test_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    model = Mlp(input_size=6, hidden_1=5, hidden_2=4, output_size=3, rng=rng)
    inputs = rng.normal(size=(3, 6))
    labels = np.array([0, 2, 1])
    class_weights = np.array([1.0, 1.3, 0.7])
    _, cache = model.forward(inputs)
    grads = model.backward(cache, labels, class_weights)

    for name, parameter in model.parameters().items():
        gradient = grads[name]
        flat_indices = rng.choice(
            parameter.size, size=min(3, parameter.size), replace=False
        )
        for flat_index in flat_indices:
            unravel = np.unravel_index(int(flat_index), parameter.shape)
            epsilon = 1e-5
            original = parameter[unravel].item()
            parameter[unravel] = original + epsilon
            loss_plus = model.loss(inputs, labels, class_weights)
            parameter[unravel] = original - epsilon
            loss_minus = model.loss(inputs, labels, class_weights)
            parameter[unravel] = original
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = gradient[unravel].item()
            assert np.isclose(numerical, analytical, rtol=1e-4, atol=1e-6), (
                f"gradient {name} did not match: analytical {analytical}, numerical {numerical}"
            )


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    model = Mlp(input_size=8, hidden_1=5, hidden_2=4, output_size=3, rng=rng)
    mean = np.array([0.1, 0.2, 0.3])
    std = np.array([1.0, 1.1, 0.9])
    class_names = ("person", "vehicle", "animal")
    path = tmp_path / "model.npz"
    save_checkpoint(path, model, mean, std, 32, class_names)

    loaded = load_checkpoint(path)

    for name, value in model.parameters().items():
        assert np.allclose(loaded.model.parameters()[name], value)
    assert np.allclose(loaded.mean, mean)
    assert np.allclose(loaded.std, std)
    assert loaded.image_size == 32
    assert loaded.class_names == class_names
