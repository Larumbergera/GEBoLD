"""User calibration: show known points, record features while the user looks at them, fine-tune.

The screen is mapped to the model output range as in the original demo (``emotional_gaze.py``):
the whole screen spans ``[-1, 1]`` in both axes, with x flipped because the model works in
camera coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gebold.features import N_FEATURES
from gebold.gaze import FineTuning, GazeModel


def grid_points(
    screen_size: tuple[int, int], cols: int = 3, rows: int = 3, margin: float = 0.05
) -> NDArray[np.float64]:
    """Calibration points in screen pixels, row by row from the top-left corner.

    Args:
        screen_size: ``(width, height)`` in pixels.
        cols: points per row.
        rows: points per column.
        margin: distance from the screen edges, as a fraction of the screen size.
    """
    width, height = screen_size
    xs = np.linspace(margin * width, (1 - margin) * width, cols)
    ys = np.linspace(margin * height, (1 - margin) * height, rows)
    return np.array([(x, y) for y in ys for x in xs])


def screen_to_target(
    points: ArrayLike, screen_size: tuple[int, int], mirror_x: bool = True
) -> NDArray[np.float64]:
    """Screen pixels to model output coordinates (inverse of :func:`gebold.inference.to_screen`)."""
    p = np.array(points, dtype=np.float64, ndmin=2)
    target = 2 * p / np.asarray(screen_size, dtype=np.float64) - 1
    if mirror_x:
        target[:, 0] *= -1
    return target


@dataclass
class CalibrationSession:
    """Collects feature vectors for each calibration point.

    Attributes:
        points: calibration points in screen pixels.
        frames_per_point: feature vectors recorded for each point.
        features: recorded vectors, grouped by point.
    """

    points: NDArray[np.float64]
    frames_per_point: int = 5
    features: list[list[NDArray]] = field(default_factory=list)

    def __post_init__(self):
        self.features = [[] for _ in range(len(self.points))]

    @property
    def index(self) -> int:
        """Index of the point being recorded (``len(points)`` when finished)."""
        for i, recorded in enumerate(self.features):
            if len(recorded) < self.frames_per_point:
                return i
        return len(self.points)

    @property
    def done(self) -> bool:
        """Whether every point has all its frames."""
        return self.index == len(self.points)

    @property
    def current_point(self) -> NDArray[np.float64] | None:
        """Point the user should look at, or ``None`` when finished."""
        return None if self.done else self.points[self.index]

    def add(self, features: ArrayLike) -> bool:
        """Record one feature vector for the current point; returns whether that point is full."""
        if self.done:
            raise RuntimeError("calibration already finished")
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (N_FEATURES,):
            raise ValueError(f"expected {N_FEATURES} features, got shape {vector.shape}")
        i = self.index
        self.features[i].append(vector)
        return len(self.features[i]) == self.frames_per_point

    def samples(
        self, screen_size: tuple[int, int], mirror_x: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Recorded features and their targets in model coordinates."""
        targets = screen_to_target(self.points, screen_size, mirror_x)
        x = [f for point in self.features for f in point]
        y = [targets[i] for i, point in enumerate(self.features) for _ in point]
        return np.array(x).reshape(-1, N_FEATURES), np.array(y).reshape(-1, 2)


def calibrate(
    model: GazeModel,
    session: CalibrationSession,
    screen_size: tuple[int, int],
    mirror_x: bool = True,
    settings: FineTuning | None = None,
) -> GazeModel:
    """Fine-tune a copy of ``model`` on the samples of a finished session.

    The calibrated model records how it was obtained in its metadata, so a saved calibration
    keeps the licence of the base weights.
    """
    if not session.done:
        raise RuntimeError("calibration not finished")
    x, y = session.samples(screen_size, mirror_x)
    calibrated = model.copy()
    history = calibrated.fine_tune(x, y, settings or FineTuning())
    calibrated.metadata |= {
        "calibration": (
            f"{len(session.points)} points x {session.frames_per_point} frames, "
            f"screen {screen_size[0]}x{screen_size[1]}, final loss {history[-1]:.4f}"
        )
    }
    return calibrated
