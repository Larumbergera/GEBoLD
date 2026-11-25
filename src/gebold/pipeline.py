"""Frame-by-frame gaze estimation: face detection, landmarks, features and point of gaze."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from gebold.detection import Face, FaceDetector
from gebold.features import features_from_face
from gebold.gaze import GazeModel
from gebold.geometry import BBox, eyes_bbox_from_eye_centres
from gebold.inference import LandmarkDetector
from gebold.models import DEFC_S, E_CLANDET, F_CLANDET, SCRFD_FILENAME, models_dir
from gebold.runtime import Device


@dataclass(frozen=True)
class FrameResult:
    """Everything computed for one frame.

    Attributes:
        face: detected face.
        eye_box: crop given to E-CLanDet.
        face_landmarks: ``(9, 2)`` F-CLanDet landmarks in frame pixels.
        eye_landmarks: ``(6, 2)`` E-CLanDet landmarks in frame pixels.
        features: the 16-value feature vector.
        pog: normalised point of gaze, in camera coordinates (see :class:`GazeModel`).
    """

    face: Face
    eye_box: BBox
    face_landmarks: NDArray[np.float64]
    eye_landmarks: NDArray[np.float64]
    features: NDArray[np.float32]
    pog: NDArray[np.float32]


class GazePipeline:
    """The thesis method with SCRFD as face detector, all in ONNX Runtime.

    As in the thesis, the eyes come from E-CLanDet and only the nose and mouth from F-CLanDet. The
    E-CLanDet crop is built from the eye centres of the detector, with the rule used for the
    thesis results; F-CLanDet gets the face box of the detector.
    """

    def __init__(
        self,
        directory: Path | None = None,
        detector_size: tuple[int, int] = (320, 320),
        device: Device = "cpu",
    ):
        directory = directory or models_dir()
        self.detector = FaceDetector(
            directory / SCRFD_FILENAME, input_size=detector_size, device=device
        )
        self.face_model = LandmarkDetector(directory / F_CLANDET.filename, device)
        self.eye_model = LandmarkDetector(directory / E_CLANDET.filename, device)
        self.gaze_model = GazeModel.load(directory / DEFC_S.filename)

    def process(self, image_bgr: NDArray[np.uint8]) -> FrameResult | None:
        """Run the whole method on a BGR frame; ``None`` if no face is found."""
        face = self.detector.detect_one(image_bgr)
        if face is None:
            return None
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        eye_box = eyes_bbox_from_eye_centres(face.keypoints[0], face.keypoints[1], width, height)
        if face.box.is_empty or eye_box.is_empty:
            return None
        face_landmarks = self.face_model.predict(rgb, face.box)
        eye_landmarks = self.eye_model.predict(rgb, eye_box)
        features = features_from_face(eye_landmarks, face_landmarks, (width, height))
        pog = self.gaze_model.predict(features)
        return FrameResult(face, eye_box, face_landmarks, eye_landmarks, features, pog)
