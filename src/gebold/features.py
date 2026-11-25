"""Feature vector fed to the gaze estimation network (thesis, section 6.3.2).

The vector has 16 values and follows ``data_preprocess.prepGaze`` of the original code, which is
what the published models were trained with. Two details differ from the thesis text and are kept
on purpose, because the trained weights depend on them:

* the x components of the pupil-to-corner vectors are absolute values;
* ``ied`` is the horizontal distance between the outer eye corners, divided by the image width.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gebold import landmarks as lm

FEATURE_NAMES: Final = (
    "right_pupil_x",
    "right_pupil_y",
    "left_pupil_x",
    "left_pupil_y",
    "d1x",  # right pupil -> right outer corner
    "d1y",
    "d2x",  # right pupil -> right caruncle
    "d2y",
    "d3x",  # left pupil -> left caruncle
    "d3y",
    "d4x",  # left pupil -> left outer corner
    "d4y",
    "ied",
    "yaw",
    "roll",
    "pitch",
)
N_FEATURES: Final = len(FEATURE_NAMES)


def pc_ec_vectors(eye: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pupil-centre to eye-corner vectors, each normalised by the size of its eye.

    Returns ``[d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y]`` with the x components in absolute value.
    """
    right_size = np.linalg.norm(eye[lm.EYE_RIGHT_INNER] - eye[lm.EYE_RIGHT_OUTER])
    left_size = np.linalg.norm(eye[lm.EYE_LEFT_OUTER] - eye[lm.EYE_LEFT_INNER])
    d = np.array(
        [
            (eye[lm.EYE_RIGHT_PUPIL] - eye[lm.EYE_RIGHT_OUTER]) / right_size,
            (eye[lm.EYE_RIGHT_PUPIL] - eye[lm.EYE_RIGHT_INNER]) / right_size,
            (eye[lm.EYE_LEFT_PUPIL] - eye[lm.EYE_LEFT_INNER]) / left_size,
            (eye[lm.EYE_LEFT_PUPIL] - eye[lm.EYE_LEFT_OUTER]) / left_size,
        ]
    )
    d[:, 0] = np.abs(d[:, 0])
    return d.ravel()


def head_pose(
    eye: NDArray[np.float64], nose: NDArray[np.float64], mouth: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Approximate head rotation from five landmarks, normalised to roughly ``[-1, 1]``.

    Args:
        eye: the six eye landmarks.
        nose: nose tip ``(x, y)``.
        mouth: centre of the mouth ``(x, y)``, i.e. the midpoint of both commissures.

    Returns:
        ``(yaw, roll, pitch)``. Yaw is the angle between the eyes-to-nose and eyes-to-mouth vectors
        in degrees divided by 90; roll is the slope of the line joining the outer eye corners in
        degrees divided by 90; pitch is the ratio of the eyes-to-nose and eyes-to-mouth distances.
    """
    right_outer, left_outer = eye[lm.EYE_RIGHT_OUTER], eye[lm.EYE_LEFT_OUTER]
    eyes_centre = (right_outer + left_outer) / 2
    to_mouth = eyes_centre - mouth
    to_nose = eyes_centre - nose

    angle = math.atan2(to_nose[1], to_nose[0]) - math.atan2(to_mouth[1], to_mouth[0])
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle <= -math.pi:
        angle += 2 * math.pi
    yaw = math.degrees(angle) / 90

    dx, dy = left_outer - right_outer
    roll = math.degrees(math.atan2(dy, dx)) / 90

    pitch = float(np.linalg.norm(to_nose) / np.linalg.norm(to_mouth))
    return yaw, roll, pitch


def gaze_features(
    eye_landmarks: ArrayLike,
    nose: ArrayLike,
    mouth_right: ArrayLike,
    mouth_left: ArrayLike,
    image_size: tuple[int, int],
) -> NDArray[np.float32]:
    """Build the 16-value feature vector of the thesis.

    Args:
        eye_landmarks: ``(6, 2)`` eye landmarks in frame pixels, in E-CLanDet order.
        nose: nose tip in frame pixels.
        mouth_right: right mouth commissure in frame pixels.
        mouth_left: left mouth commissure in frame pixels.
        image_size: ``(width, height)`` of the frame.

    Returns:
        ``float32`` array ordered as :data:`FEATURE_NAMES`.
    """
    eye = np.asarray(eye_landmarks, dtype=np.float64)
    if eye.shape != (lm.N_EYE_LANDMARKS, 2):
        raise ValueError(f"expected eye landmarks of shape (6, 2), got {eye.shape}")
    width, height = image_size
    mouth = (
        np.asarray(mouth_right, dtype=np.float64) + np.asarray(mouth_left, dtype=np.float64)
    ) / 2

    right_pupil = eye[lm.EYE_RIGHT_PUPIL] / (width, height)
    left_pupil = eye[lm.EYE_LEFT_PUPIL] / (width, height)
    ied = abs(eye[lm.EYE_LEFT_OUTER, 0] - eye[lm.EYE_RIGHT_OUTER, 0]) / width
    yaw, roll, pitch = head_pose(eye, np.asarray(nose, dtype=np.float64), mouth)

    return np.array(
        [*right_pupil, *left_pupil, *pc_ec_vectors(eye), ied, yaw, roll, pitch], dtype=np.float32
    )


def features_from_face(
    eye_landmarks: ArrayLike, face_landmarks: ArrayLike, image_size: tuple[int, int]
) -> NDArray[np.float32]:
    """Feature vector from E-CLanDet's eye landmarks and F-CLanDet's facial landmarks.

    As in the thesis, the eyes come from E-CLanDet and only the nose and mouth from F-CLanDet.
    """
    face = np.asarray(face_landmarks, dtype=np.float64)
    if face.shape != (lm.N_FACE_LANDMARKS, 2):
        raise ValueError(f"expected face landmarks of shape (9, 2), got {face.shape}")
    return gaze_features(
        eye_landmarks,
        face[lm.FACE_NOSE_TIP],
        face[lm.FACE_MOUTH_RIGHT],
        face[lm.FACE_MOUTH_LEFT],
        image_size,
    )
