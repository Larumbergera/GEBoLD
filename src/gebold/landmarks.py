"""Landmark layouts produced by the two landmark detectors of the thesis.

Both detectors output landmarks in image order (the user's right eye appears on the left of the
image). Indices come from the iBUG 68-point mark-up extended with the two pupil centres (68 and
69), as in ``utils_load.getEyeIndexes`` of the original code.
"""

from typing import Final

# E-CLanDet: six eye landmarks, iBUG points [36, 68, 39, 42, 69, 45].
EYE_RIGHT_OUTER: Final = 0
EYE_RIGHT_PUPIL: Final = 1
EYE_RIGHT_INNER: Final = 2  # caruncle
EYE_LEFT_INNER: Final = 3  # caruncle
EYE_LEFT_PUPIL: Final = 4
EYE_LEFT_OUTER: Final = 5
N_EYE_LANDMARKS: Final = 6

# F-CLanDet: nine facial landmarks, iBUG points [30, 36, 39, 42, 45, 48, 54, 68, 69].
FACE_NOSE_TIP: Final = 0
FACE_RIGHT_OUTER: Final = 1
FACE_RIGHT_INNER: Final = 2
FACE_LEFT_INNER: Final = 3
FACE_LEFT_OUTER: Final = 4
FACE_MOUTH_RIGHT: Final = 5
FACE_MOUTH_LEFT: Final = 6
FACE_RIGHT_PUPIL: Final = 7
FACE_LEFT_PUPIL: Final = 8
N_FACE_LANDMARKS: Final = 9
