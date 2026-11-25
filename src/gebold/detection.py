"""Face detection with SCRFD (InsightFace) through ONNX Runtime.

SCRFD replaces the MTCNN detector used in the thesis: on CPU, SCRFD-500MF at 320x320 takes a few
milliseconds per frame while MTCNN took about half a second, so the face can be detected on every
frame instead of tracking it from the previous landmarks.

The decoding follows ``insightface/model_zoo/scrfd.py`` (MIT licence). The pretrained weights are
released by InsightFace for non-commercial research only and are downloaded, not bundled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from gebold.geometry import BBox
from gebold.runtime import Device, session

STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2


@dataclass(frozen=True)
class Face:
    """A detected face.

    Attributes:
        box: face box in frame pixels.
        score: detection confidence.
        keypoints: ``(5, 2)`` points in image order: eye on the left of the image (the user's
            right eye), eye on the right of the image, nose tip, left and right mouth corners.
    """

    box: BBox
    score: float
    keypoints: NDArray[np.float64]


def letterbox(image: NDArray[np.uint8], size: tuple[int, int]) -> tuple[NDArray[np.uint8], float]:
    """Resize keeping the aspect ratio into the top-left corner of a black ``size`` canvas.

    Args:
        image: BGR image.
        size: ``(width, height)`` of the network input.

    Returns:
        The canvas and the scale applied to the image.
    """
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h))
    return canvas, scale


def anchor_centres(height: int, width: int, stride: int) -> NDArray[np.float32]:
    """Centres of the anchors of one feature map, repeated for each anchor of a cell."""
    grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    return np.repeat((grid * stride).reshape(-1, 2), ANCHORS_PER_CELL, axis=0)


def nms(boxes: NDArray, scores: NDArray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression over ``[x1, y1, x2, y2]`` boxes; returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        iou = w * h / (areas[i] + areas[rest] - w * h)
        order = rest[iou <= iou_threshold]
    return keep


def decode(
    outputs: Sequence[NDArray],
    input_size: tuple[int, int],
    score_threshold: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Turn the nine SCRFD outputs into boxes, scores and keypoints in input pixels.

    Args:
        outputs: scores, box distances and keypoint distances for strides 8, 16 and 32, in that
            order (the layout of the InsightFace ONNX exports), with or without a batch axis.
        input_size: ``(width, height)`` of the network input.
        score_threshold: minimum score to keep a candidate.

    Returns:
        ``(boxes [N, 4] as x1, y1, x2, y2; scores [N]; keypoints [N, 5, 2])``.
    """
    outs = [o[0] if o.ndim == 3 else o for o in outputs]
    width, height = input_size
    n = len(STRIDES)
    boxes, scores, kpss = [], [], []
    for i, stride in enumerate(STRIDES):
        score = outs[i][:, 0]
        keep = score >= score_threshold
        centres = anchor_centres(height // stride, width // stride, stride)[keep]
        dist = outs[i + n][keep] * stride
        kps = outs[i + 2 * n][keep] * stride
        boxes.append(np.hstack([centres - dist[:, :2], centres + dist[:, 2:]]))
        kpss.append((np.tile(centres, 5) + kps).reshape(-1, 5, 2))
        scores.append(score[keep])
    return np.vstack(boxes), np.concatenate(scores), np.vstack(kpss)


class FaceDetector:
    """SCRFD face detector.

    Args:
        model_path: SCRFD ONNX file with keypoints (e.g. ``det_500m.onnx``).
        input_size: ``(width, height)`` of the network input. 320x320 is enough for a user in
            front of a webcam.
        score_threshold: minimum confidence.
        iou_threshold: overlap above which duplicate detections are removed.
        device: ``cpu`` or ``cuda``.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        device: Device = "cpu",
    ):
        self.session = session(model_path, device)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[Face]:
        """Detect faces in a BGR frame, most confident first."""
        canvas, scale = letterbox(image_bgr, self.input_size)
        blob = ((canvas[..., ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outputs = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})
        boxes, scores, kpss = decode(outputs, self.input_size, self.score_threshold)
        if not len(scores):
            return []
        keep = nms(boxes, scores, self.iou_threshold)
        height, width = image_bgr.shape[:2]
        faces = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i] / scale
            box = BBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1)).clip(width, height)
            faces.append(Face(box, float(scores[i]), kpss[i].astype(np.float64) / scale))
        return faces

    def detect_one(self, image_bgr: NDArray[np.uint8]) -> Face | None:
        """The most confident face, or ``None``."""
        faces = self.detect(image_bgr)
        return faces[0] if faces else None
