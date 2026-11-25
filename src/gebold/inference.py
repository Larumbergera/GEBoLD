"""Inference with the thesis landmark models exported to ONNX (no TensorFlow needed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gebold.geometry import BBox
from gebold.runtime import Device, session


def normalise_image(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Scale RGB values from ``[0, 255]`` to ``[-1, 1]``, as in training (``utils_load``)."""
    return (image_rgb.astype(np.float32) - 127.5) / 127.5


class LandmarkDetector:
    """E-CLanDet or F-CLanDet: a ResNet50 regressing landmark coordinates from an image crop.

    The crop is resized to the model input with ``INTER_AREA`` and normalised to ``[-1, 1]``, as
    in ``data_preprocess.prepEye``/``prepFace``; the model returns ``(x, y)`` pixels of the resized
    crop, which are mapped back to the frame.
    """

    def __init__(self, model_path: str | Path, device: Device = "cpu"):
        self.session = session(model_path, device)
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        self.input_name = inp.name
        self.input_size: tuple[int, int] = (int(inp.shape[1]), int(inp.shape[2]))  # (h, w)
        self.n_landmarks = int(out.shape[1]) // 2

    def predict(self, image_rgb: NDArray[np.uint8], box: BBox) -> NDArray[np.float64]:
        """Landmarks ``(N, 2)`` in frame pixels for the region ``box`` of an RGB frame.

        Raises:
            ValueError: if the box is empty.
        """
        if box.is_empty:
            raise ValueError(f"empty box: {box}")
        out_h, out_w = self.input_size
        crop = cv2.resize(box.crop(image_rgb), (out_w, out_h), interpolation=cv2.INTER_AREA)
        pred = self.session.run(None, {self.input_name: normalise_image(crop)[None]})[0]
        return box.from_crop(pred.reshape(self.n_landmarks, 2), self.input_size)


def to_screen(
    pog: ArrayLike, screen_size: tuple[int, int], mirror_x: bool = True
) -> NDArray[np.float64]:
    """Normalised point of gaze to screen pixels.

    Args:
        pog: ``(x, y)`` in ``[-1, 1]`` as returned by :class:`gebold.gaze.GazeModel`.
        screen_size: ``(width, height)`` in pixels.
        mirror_x: flip x, because the model works in camera coordinates (x grows to the user's
            left) while screen pixels grow to the user's right. Checked on I2Head: predicted x
            correlates +0.98 with camera x. Disable it for webcams that deliver mirrored frames.
    """
    x, y = np.asarray(pog, dtype=np.float64)
    if mirror_x:
        x = -x
    width, height = screen_size
    return np.array([(x + 1) / 2 * width, (y + 1) / 2 * height])
