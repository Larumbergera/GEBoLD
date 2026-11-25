"""DEFC gaze model in numpy: inference and per-user calibration by fine-tuning.

DEFC is a small fully connected network (16 -> 32 -> 16 -> 16 -> 8 -> 8 -> 2, ELU hidden layers,
linear output; about 1,600 weights), so it is evaluated and fine-tuned in numpy. The calibration
reproduces the thesis code (``finetune_gaze_session_phd.py``): Keras' Adam, every layer
trainable, and the mean Euclidean distance between predicted and true point of gaze as loss
(``losses.pog_euc``; the thesis text says squared error, the code does not square it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gebold.features import N_FEATURES


@dataclass(frozen=True)
class FineTuning:
    """Fine-tuning settings.

    The defaults are the user calibration of the thesis (model DEFC-S-C, calibrated with the
    17-point session of each user): Adam with lr = 6e-3 * batch / 128, batch 1, 200 epochs.
    """

    epochs: int = 200
    batch_size: int = 1
    lr: float = 6e-3 / 128
    shuffle: bool = True
    seed: int | None = 0


def elu(x: NDArray) -> NDArray:
    """Exponential linear unit with alpha 1 (Keras' ``elu``)."""
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def elu_grad(x: NDArray) -> NDArray:
    """Derivative of :func:`elu` with respect to its input."""
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))


@dataclass
class Adam:
    """Keras' Adam (``beta_1=0.9``, ``beta_2=0.999``, ``epsilon=1e-7``)."""

    lr: float
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    step: int = 0
    m: list[NDArray] = field(default_factory=list)
    v: list[NDArray] = field(default_factory=list)

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        """Apply one step in place."""
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.step += 1
        alpha = self.lr * np.sqrt(1 - self.beta_2**self.step) / (1 - self.beta_1**self.step)
        for i, g in enumerate(grads):
            self.m[i] += (g - self.m[i]) * (1 - self.beta_1)
            self.v[i] += (g * g - self.v[i]) * (1 - self.beta_2)
            params[i] -= alpha * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)


def pog_loss(pred: NDArray, true: NDArray) -> tuple[float, NDArray]:
    """Mean Euclidean distance between points of gaze, and its gradient w.r.t. ``pred``."""
    diff = pred - true
    dist = np.sqrt((diff**2).sum(axis=1, keepdims=True))
    grad = diff / np.maximum(dist, 1e-12) / len(pred)
    return float(dist.mean()), grad


class GazeModel:
    """DEFC network.

    The output is the point of gaze normalised to ``[-1, 1]`` over the training grid, in camera
    coordinates: positive x points to the user's left. :func:`gebold.inference.to_screen`
    converts it to pixels.

    Args:
        weights: kernels and biases of each dense layer, ``[W0, b0, W1, b1, ...]``.
        metadata: free-form strings saved with the weights (licence, origin, calibration).
    """

    def __init__(self, weights: list[NDArray], metadata: dict[str, str] | None = None):
        if len(weights) % 2 or weights[0].shape[0] != N_FEATURES:
            raise ValueError("weights must be [W0, b0, W1, b1, ...] with W0 of 16 rows")
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path: str | Path) -> GazeModel:
        """Load a model saved with :meth:`save`."""
        with np.load(path) as data:
            n = sum(1 for k in data.files if k.startswith("w"))
            weights = [data[f"{p}{i}"] for i in range(n) for p in ("w", "b")]
            meta = {k.removeprefix("meta_"): str(data[k]) for k in data.files if "meta_" in k}
        return cls(weights, meta)

    def save(self, path: str | Path) -> None:
        """Save weights and metadata to an ``.npz`` file."""
        arrays = {f"{'wb'[i % 2]}{i // 2}": w for i, w in enumerate(self.weights)}
        arrays |= {f"meta_{k}": np.array(v) for k, v in self.metadata.items()}
        np.savez(path, **arrays)

    def copy(self) -> GazeModel:
        """Independent copy, e.g. to calibrate without touching the base model."""
        return GazeModel([w.copy() for w in self.weights], self.metadata)

    def _forward(self, x: NDArray) -> tuple[NDArray, list[NDArray], list[NDArray]]:
        inputs, pre = [], []
        h = x
        n_layers = len(self.weights) // 2
        for i in range(n_layers):
            inputs.append(h)
            z = h @ self.weights[2 * i] + self.weights[2 * i + 1]
            pre.append(z)
            h = elu(z) if i < n_layers - 1 else z
        return h, inputs, pre

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Normalised point of gaze ``(x, y)`` for one feature vector, or ``(N, 2)`` for N."""
        x = np.asarray(features, dtype=np.float64)
        if x.shape[-1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {x.shape[-1]}")
        out = self._forward(np.atleast_2d(x))[0]
        return out[0] if x.ndim == 1 else out

    def train_step(self, x: NDArray, y: NDArray, optimizer: Adam) -> float:
        """One gradient step on a batch; returns the loss before the step."""
        out, inputs, pre = self._forward(x)
        loss, grad = pog_loss(out, y)
        grads: list[NDArray] = [np.empty(0)] * len(self.weights)
        for i in reversed(range(len(inputs))):
            if i < len(inputs) - 1:
                grad = grad * elu_grad(pre[i])
            grads[2 * i] = inputs[i].T @ grad
            grads[2 * i + 1] = grad.sum(axis=0)
            grad = grad @ self.weights[2 * i].T
        optimizer.update(self.weights, grads)
        return loss

    def fine_tune(
        self, features: ArrayLike, targets: ArrayLike, settings: FineTuning | None = None
    ) -> list[float]:
        """Calibrate in place on user samples; returns the mean loss of each epoch.

        Args:
            features: ``(N, 16)`` feature vectors recorded while the user looked at known points.
            targets: ``(N, 2)`` normalised points of gaze of those points (camera coordinates).
            settings: optimiser settings; the defaults reproduce the thesis calibration.
        """
        settings = settings or FineTuning()
        x = np.asarray(features, dtype=np.float64).reshape(-1, N_FEATURES)
        y = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        if len(x) != len(y) or not len(x):
            raise ValueError("need the same, non-zero number of features and targets")
        rng = np.random.default_rng(settings.seed)
        optimizer = Adam(settings.lr)
        history = []
        bs = settings.batch_size
        for _ in range(settings.epochs):
            order = rng.permutation(len(x)) if settings.shuffle else np.arange(len(x))
            batches = [order[i : i + bs] for i in range(0, len(x), bs)]
            losses = [self.train_step(x[idx], y[idx], optimizer) for idx in batches]
            history.append(float(np.mean(losses)))
        return history
