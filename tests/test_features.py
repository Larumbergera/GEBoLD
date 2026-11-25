import math

import numpy as np
import pytest

from gebold import landmarks as lm
from gebold.features import (
    FEATURE_NAMES,
    N_FEATURES,
    features_from_face,
    gaze_features,
    head_pose,
    pc_ec_vectors,
)

W, H = 640, 480


def frontal_face():
    """A symmetric, upright face looking straight ahead."""
    eye = np.array(
        [
            [250.0, 200.0],  # right outer
            [270.0, 200.0],  # right pupil
            [290.0, 200.0],  # right caruncle
            [350.0, 200.0],  # left caruncle
            [370.0, 200.0],  # left pupil
            [390.0, 200.0],  # left outer
        ]
    )
    nose = np.array([320.0, 260.0])
    mouth_right = np.array([290.0, 320.0])
    mouth_left = np.array([350.0, 320.0])
    return eye, nose, mouth_right, mouth_left


def original_prep_gaze(eye, face9, width, height):
    """Transcription of ``data_preprocess.prepGaze`` (thesis code, 2023) used as reference."""
    ied = abs(eye[5, 0] - eye[0, 0]) / width
    rs = np.sqrt(np.sum(np.square(eye[2] - eye[0])))
    ls = np.sqrt(np.sum(np.square(eye[5] - eye[3])))
    d1 = (eye[1] - eye[0]) / rs
    d2 = (eye[1] - eye[2]) / rs
    d3 = (eye[4] - eye[3]) / ls
    d4 = (eye[4] - eye[5]) / ls
    facial = face9[[0, 5, 6]]  # nose and both commissures
    p1 = (eye[5] + eye[0]) / 2
    p2, p3 = facial[0], (facial[1] + facial[2]) / 2
    v1, v2 = p1 - p3, p1 - p2
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    yaw = np.degrees(angle) / 90
    dif = eye[5] - eye[0]
    roll = np.degrees(np.arctan2(dif[1], dif[0])) / 90
    pitch = np.sqrt(np.sum(np.square(p1 - p2))) / np.sqrt(np.sum(np.square(p1 - p3)))
    return [
        eye[1, 0] / width, eye[1, 1] / height, eye[4, 0] / width, eye[4, 1] / height,
        abs(d1[0]), d1[1], abs(d2[0]), d2[1], abs(d3[0]), d3[1], abs(d4[0]), d4[1],
        ied, yaw, roll, pitch,
    ]  # fmt: skip


def test_names_match_length():
    assert len(FEATURE_NAMES) == N_FEATURES == 16


def test_frontal_face_has_no_rotation():
    eye, nose, mr, ml = frontal_face()
    yaw, roll, pitch = head_pose(eye, nose, (mr + ml) / 2)
    assert yaw == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(60 / 120)


def test_pc_ec_vectors_centred_pupils():
    eye, *_ = frontal_face()
    assert pc_ec_vectors(eye) == pytest.approx([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0])


def test_roll_follows_eye_line():
    eye, nose, mr, ml = frontal_face()
    eye[lm.EYE_LEFT_OUTER, 1] += 140  # left corner 140 px lower: 45 degrees
    _, roll, _ = head_pose(eye, nose, (mr + ml) / 2)
    assert roll == pytest.approx(math.degrees(math.atan2(140, 140)) / 90)


def test_full_vector_values():
    eye, nose, mr, ml = frontal_face()
    f = gaze_features(eye, nose, mr, ml, (W, H))
    assert f.dtype == np.float32
    named = dict(zip(FEATURE_NAMES, f, strict=True))
    assert named["right_pupil_x"] == pytest.approx(270 / W)
    assert named["left_pupil_y"] == pytest.approx(200 / H)
    assert named["ied"] == pytest.approx(140 / W)


def test_uses_both_mouth_commissures():
    # The 2022 demo averaged the left commissure with itself; the mouth centre must use both.
    eye, nose, mr, ml = frontal_face()
    good = gaze_features(eye, nose, mr, ml, (W, H))
    buggy = gaze_features(eye, nose, ml, ml, (W, H))
    yaw = FEATURE_NAMES.index("yaw")
    assert good[yaw] == pytest.approx(0.0)
    assert buggy[yaw] != pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(20))
def test_matches_original_prep_gaze(seed):
    rng = np.random.default_rng(seed)
    eye, nose, mr, ml = frontal_face()
    face9 = np.zeros((lm.N_FACE_LANDMARKS, 2))
    face9[lm.FACE_NOSE_TIP], face9[lm.FACE_MOUTH_RIGHT], face9[lm.FACE_MOUTH_LEFT] = nose, mr, ml
    eye = eye + rng.normal(0, 8, eye.shape)
    face9 = face9 + rng.normal(0, 8, face9.shape)
    expected = original_prep_gaze(eye, face9, W, H)
    got = features_from_face(eye, face9, (W, H))
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-6, atol=1e-7)


def test_rejects_wrong_shapes():
    eye, *_ = frontal_face()
    with pytest.raises(ValueError, match=r"\(6, 2\)"):
        gaze_features(eye[:5], [0, 0], [0, 0], [0, 0], (W, H))
    with pytest.raises(ValueError, match=r"\(9, 2\)"):
        features_from_face(eye, np.zeros((14, 2)), (W, H))
