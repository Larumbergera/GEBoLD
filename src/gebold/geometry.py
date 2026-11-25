"""Bounding boxes and coordinate changes between the full frame and the detector crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gebold import landmarks as lm


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixels: top-left corner ``(x, y)``, width ``w`` and height ``h``."""

    x: int
    y: int
    w: int
    h: int

    @property
    def is_empty(self) -> bool:
        """Whether the box has no area."""
        return self.w <= 0 or self.h <= 0

    def clip(self, width: int, height: int) -> BBox:
        """Clip the box to an image of ``width`` x ``height`` pixels.

        The original code moved the corner to 0 without shrinking the box, so a box sticking out
        of the image on the left or top kept its full width or height. Here both ends are clipped.
        """
        x0, y0 = max(self.x, 0), max(self.y, 0)
        x1, y1 = min(self.x + self.w, width), min(self.y + self.h, height)
        return BBox(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))

    def crop(self, image: NDArray) -> NDArray:
        """Return the part of ``image`` inside the box (the box must already be clipped)."""
        return image[self.y : self.y + self.h, self.x : self.x + self.w]

    def to_crop(self, points: ArrayLike, out_size: tuple[int, int]) -> NDArray[np.float64]:
        """Map frame coordinates to the coordinates of the crop resized to ``out_size``.

        Args:
            points: ``(N, 2)`` array of ``(x, y)`` points in the frame.
            out_size: ``(height, width)`` of the resized crop, as in the models' ``in_dim``.
        """
        p = np.array(points, dtype=np.float64, ndmin=2)
        out_h, out_w = out_size
        p[:, 0] = (p[:, 0] - self.x) * out_w / self.w
        p[:, 1] = (p[:, 1] - self.y) * out_h / self.h
        return p

    def from_crop(self, points: ArrayLike, out_size: tuple[int, int]) -> NDArray[np.float64]:
        """Map coordinates of the resized crop back to the frame (inverse of :meth:`to_crop`)."""
        p = np.array(points, dtype=np.float64, ndmin=2)
        out_h, out_w = out_size
        p[:, 0] = p[:, 0] * self.w / out_w + self.x
        p[:, 1] = p[:, 1] * self.h / out_h + self.y
        return p


def _box(x0: float, y0: float, w: float, h: float, width: int, height: int) -> BBox:
    return BBox(int(x0), int(y0), int(w), int(h)).clip(width, height)


def face_bbox(points: ArrayLike, width: int, height: int, factor: float = 0.25) -> BBox:
    """Box around a set of facial landmarks, enlarged by ``factor`` and clipped to the image.

    Used by F-CLanDet: in the thesis the face box spans the nine landmarks and is enlarged by 1.25.
    """
    p = np.asarray(points, dtype=np.float64)
    x_min, y_min = p.min(axis=0)
    x_max, y_max = p.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    dw, dh = w * factor, h * factor
    return _box(x_min - dw / 2, y_min - dh / 2, w + dw, h + dh, width, height)


def eyes_bbox(
    right_outer: ArrayLike,
    left_outer: ArrayLike,
    width: int,
    height: int,
    factor: float = 0.0,
) -> BBox:
    """Box spanning both eyes from the two outer eye corners, as used to crop E-CLanDet's input.

    The box is at least half as tall as it is wide (centred on the corners) and is then enlarged by
    ``factor`` in both directions.
    """
    xr, yr = np.asarray(right_outer, dtype=np.float64)
    xl, yl = np.asarray(left_outer, dtype=np.float64)
    x, y = min(xr, xl), min(yr, yl)
    w, h = abs(xl - xr), abs(yl - yr)
    if 0.5 * w > h:
        y -= (0.5 * w - h) / 2
        h = 0.5 * w
    dw, dh = w * factor, h * factor
    return _box(x - dw / 2, y - dh / 2, w + dw, h + dh, width, height)


def eyes_bbox_from_eye_centres(
    eye_a: ArrayLike, eye_b: ArrayLike, width: int, height: int, rate: float = 7 / 5
) -> BBox:
    """E-CLanDet input box from the two eye centres given by the face detector.

    Same rule as ``createbbox`` in the original ``mtcnn_databases.py``, which produced the crops
    behind the thesis results: the box spans both eye centres plus 0.4 times their distance on
    each side, is at least half as tall as it is wide, and gets 10 % more height.
    """
    (xa, ya), (xb, yb) = np.asarray(eye_a, dtype=np.float64), np.asarray(eye_b, dtype=np.float64)
    w, h = rate * abs(xb - xa), rate * abs(yb - ya)
    x0, x1 = (xb - w, xa + w) if xa <= xb else (xa - w, xb + w)
    y = min(ya, yb)
    if 0.5 * w > h:
        y -= (0.5 * w - h) / 2
        h = 0.5 * w
    dh = 0.1 * h
    return _box(x0, y - dh / 2, x1 - x0, h + dh, width, height)


def eyes_bbox_from_face(face_landmarks: ArrayLike, width: int, height: int, factor: float) -> BBox:
    """Eye box computed from F-CLanDet's outer eye corners."""
    f = np.asarray(face_landmarks)
    return eyes_bbox(f[lm.FACE_RIGHT_OUTER], f[lm.FACE_LEFT_OUTER], width, height, factor)
