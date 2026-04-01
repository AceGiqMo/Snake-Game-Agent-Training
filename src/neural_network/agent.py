from dataclasses import dataclass
from typing import Optional

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    # Numerically stable softmax
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


@dataclass(frozen=True)
class NetworkSpec:
    input_size: int
    hidden1: int = 256  # 128-256
    hidden2: int = 128  # 64-128
    output_size: int = 4

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        if not (128 <= self.hidden1 <= 256):
            raise ValueError("hidden1 must be in [128, 256].")
        if not (64 <= self.hidden2 <= 128):
            raise ValueError("hidden2 must be in [64, 128].")
        if self.output_size != 4:
            raise ValueError("output_size must be 4 (up/down/left/right).")


class MLP:
    """
    NumPy-only feed-forward neural network.

    Architecture:
      input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax(4)

    Note: This is intentionally forward-only; PSO will optimize weights externally.
    """

    def __init__(self, spec: NetworkSpec, rng: Optional[np.random.Generator] = None):
        self.spec = spec
        self.rng = rng or np.random.default_rng()

        # Xavier/Glorot-like init for ReLU layers: N(0, sqrt(2/fan_in))
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / spec.input_size), size=(spec.input_size, spec.hidden1)).astype(
            np.float32
        )
        self.b1 = np.zeros((spec.hidden1,), dtype=np.float32)

        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / spec.hidden1), size=(spec.hidden1, spec.hidden2)).astype(
            np.float32
        )
        self.b2 = np.zeros((spec.hidden2,), dtype=np.float32)

        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / spec.hidden2), size=(spec.hidden2, spec.output_size)).astype(
            np.float32
        )
        self.b3 = np.zeros((spec.output_size,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        - x shape: (input_size,) or (batch, input_size)
        - returns shape: (4,) or (batch, 4) probabilities
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            if x.shape[0] != self.spec.input_size:
                raise ValueError(f"Expected input of shape ({self.spec.input_size},), got {tuple(x.shape)}.")
        elif x.ndim == 2:
            if x.shape[1] != self.spec.input_size:
                raise ValueError(f"Expected input of shape (batch, {self.spec.input_size}), got {tuple(x.shape)}.")
        else:
            raise ValueError("Input must be 1D or 2D array.")

        z1 = x @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return _softmax(z3)

    def predict_action(self, x: np.ndarray) -> int:
        """Returns action index in {0,1,2,3} = (UP, DOWN, LEFT, RIGHT)."""
        probs = self.forward(x)
        if probs.ndim == 2:
            return int(np.argmax(probs, axis=1)[0])
        return int(np.argmax(probs))

    def num_parameters(self) -> int:
        return (
            self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
            + self.W3.size
            + self.b3.size
        )

    def get_parameters_vector(self) -> np.ndarray:
        """Flattens all weights/biases into one 1D vector (for PSO)."""
        return np.concatenate(
            [
                self.W1.ravel(),
                self.b1.ravel(),
                self.W2.ravel(),
                self.b2.ravel(),
                self.W3.ravel(),
                self.b3.ravel(),
            ]
        ).astype(np.float32, copy=False)

    def set_parameters_vector(self, vec: np.ndarray) -> None:
        """Loads all weights/biases from one 1D vector (from PSO)."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        expected = self.num_parameters()
        if vec.size != expected:
            raise ValueError(f"Parameter vector length mismatch: expected {expected}, got {vec.size}.")

        i = 0
        w1_n = self.W1.size
        self.W1 = vec[i : i + w1_n].reshape(self.W1.shape)
        i += w1_n

        b1_n = self.b1.size
        self.b1 = vec[i : i + b1_n].reshape(self.b1.shape)
        i += b1_n

        w2_n = self.W2.size
        self.W2 = vec[i : i + w2_n].reshape(self.W2.shape)
        i += w2_n

        b2_n = self.b2.size
        self.b2 = vec[i : i + b2_n].reshape(self.b2.shape)
        i += b2_n

        w3_n = self.W3.size
        self.W3 = vec[i : i + w3_n].reshape(self.W3.shape)
        i += w3_n

        b3_n = self.b3.size
        self.b3 = vec[i : i + b3_n].reshape(self.b3.shape)


class SnakeAgent:
    """
    High-level agent wrapper around the MLP.

    - Reads input size from `nn_architecture.csv` (unless `input_size` is provided).
    - Exposes `act(observation)` -> direction index 0..3.
    """

    ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

    def __init__(
        self,
        *,
        input_size: int = 23,
        hidden1: int = 256,
        hidden2: int = 128,
        rng: Optional[np.random.Generator] = None,
    ):

        self.spec = NetworkSpec(input_size=input_size, hidden1=hidden1, hidden2=hidden2, output_size=4)
        self.net = MLP(self.spec, rng=rng)

    def act(self, observation: np.ndarray) -> int:
        """Returns 0..3 for (UP, DOWN, LEFT, RIGHT)."""
        return SnakeAgent.ACTIONS[self.net.predict_action(observation)]

    def action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Returns softmax probabilities of length 4."""
        p = self.net.forward(observation)
        if p.ndim == 2:
            return p[0]
        return p

    def num_parameters(self) -> int:
        return self.net.num_parameters()

    def get_weights(self) -> np.ndarray:
        """Alias for PSO: returns 1D weights vector."""
        return self.net.get_parameters_vector()

    def set_weights(self, weights: np.ndarray) -> None:
        """Alias for PSO: loads from 1D weights vector."""
        self.net.set_parameters_vector(weights)

