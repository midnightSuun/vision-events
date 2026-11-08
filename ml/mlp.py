"""Трёхслойный классификатор на NumPy."""

from pathlib import Path

import numpy as np

BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray, class_weights: np.ndarray) -> float:
    chosen = probabilities[np.arange(labels.shape[0]), labels]
    weighted = class_weights[labels] * np.log(np.clip(chosen, 1e-12, 1.0))
    return float(-np.mean(weighted))


class Mlp:
    def __init__(self, input_size: int, hidden_1: int, hidden_2: int, output_size: int, rng: np.random.Generator) -> None:
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden_1))
        self.b1 = np.zeros(hidden_1)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_1), size=(hidden_1, hidden_2))
        self.b2 = np.zeros(hidden_2)
        self.w3 = rng.normal(0.0, np.sqrt(1.0 / hidden_2), size=(hidden_2, output_size))
        self.b3 = np.zeros(output_size)

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
        }

    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters().values())

    def restore(self, saved: dict[str, np.ndarray]) -> None:
        for name, value in self.parameters().items():
            value[...] = saved[name]

    def clone_parameters(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.parameters().items()}

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = inputs @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        logits = a2 @ self.w3 + self.b3
        probabilities = softmax(logits)
        cache = {
            "inputs": inputs,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "probabilities": probabilities,
        }
        return probabilities, cache

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        probabilities, _ = self.forward(inputs)
        return probabilities

    def loss(self, inputs: np.ndarray, labels: np.ndarray, class_weights: np.ndarray) -> float:
        probabilities, _ = self.forward(inputs)
        return cross_entropy(probabilities, labels, class_weights)

    def backward(self, cache: dict[str, np.ndarray], labels: np.ndarray, class_weights: np.ndarray) -> dict[str, np.ndarray]:
        inputs = cache["inputs"]
        a1 = cache["a1"]
        a2 = cache["a2"]
        probabilities = cache["probabilities"]
        batch_size = probabilities.shape[0]
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(batch_size), labels] = 1.0
        sample_weight = class_weights[labels][:, np.newaxis]
        dz3 = sample_weight * (probabilities - one_hot) / batch_size

        dw3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)
        dz2 = (dz3 @ self.w3.T) * (cache["z2"] > 0.0)
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.w2.T) * (cache["z1"] > 0.0)
        dw1 = inputs.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}


def verify_gradients(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    model = Mlp(input_size=6, hidden_1=5, hidden_2=4, output_size=3, rng=rng)
    inputs = rng.normal(size=(3, 6))
    labels = np.array([0, 2, 1])
    class_weights = np.array([1.0, 1.3, 0.7])
    _, cache = model.forward(inputs)
    grads = model.backward(cache, labels, class_weights)

    for name, parameter in model.parameters().items():
        gradient = grads[name]
        flat_indices = rng.choice(parameter.size, size=min(3, parameter.size), replace=False)
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
            if not np.isclose(numerical, analytical, rtol=1e-4, atol=1e-6):
                raise RuntimeError(f"градиент {name} не сошёлся: аналитический {analytical}, численный {numerical}")


def new_adam_state(model: Mlp) -> dict[str, dict[str, np.ndarray | float]]:
    state: dict[str, dict[str, np.ndarray | float]] = {}
    for name, value in model.parameters().items():
        decay = 0.0 if name.startswith("b") else 1.0
        state[name] = {"m": np.zeros_like(value), "v": np.zeros_like(value), "decay": decay}
    return state


def adam_step(
    model: Mlp,
    grads: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray | float]],
    step: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    clip_norm = 5.0
    total_norm = float(np.sqrt(sum(np.square(grad).sum() for grad in grads.values())))
    scale = 1.0 if total_norm <= clip_norm else clip_norm / (total_norm + 1e-12)

    for name, parameter in model.parameters().items():
        moment = state[name]["m"]
        velocity = state[name]["v"]
        decay = float(state[name]["decay"]) * weight_decay
        grad = grads[name] * scale
        moment *= BETA1
        moment += (1.0 - BETA1) * grad
        velocity *= BETA2
        velocity += (1.0 - BETA2) * np.square(grad)
        moment_hat = moment / (1.0 - BETA1**step)
        velocity_hat = velocity / (1.0 - BETA2**step)
        parameter -= learning_rate * (moment_hat / (np.sqrt(velocity_hat) + ADAM_EPS) + decay * parameter)


def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "model.npz"


def save_checkpoint(
    path: Path,
    model: Mlp,
    mean: np.ndarray,
    std: np.ndarray,
    image_size: int,
    class_names: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = model.parameters()
    np.savez(
        path,
        w1=parameters["w1"],
        b1=parameters["b1"],
        w2=parameters["w2"],
        b2=parameters["b2"],
        w3=parameters["w3"],
        b3=parameters["b3"],
        mean=mean,
        std=std,
        image_size=np.int64(image_size),
        class_names=np.array("|".join(class_names)),
    )


def load_checkpoint(path: Path) -> tuple[Mlp, np.ndarray, np.ndarray, int, tuple[str, ...]]:
    with np.load(path) as checkpoint:
        model = Mlp.__new__(Mlp)
        model.w1 = checkpoint["w1"].copy()
        model.b1 = checkpoint["b1"].copy()
        model.w2 = checkpoint["w2"].copy()
        model.b2 = checkpoint["b2"].copy()
        model.w3 = checkpoint["w3"].copy()
        model.b3 = checkpoint["b3"].copy()
        mean = checkpoint["mean"].copy()
        std = checkpoint["std"].copy()
        image_size = int(checkpoint["image_size"])
        class_names = tuple(str(checkpoint["class_names"].item()).split("|"))
    return model, mean, std, image_size, class_names
