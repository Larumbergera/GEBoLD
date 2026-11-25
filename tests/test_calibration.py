from itertools import pairwise

import numpy as np
import pytest

from gebold.calibration import CalibrationSession, calibrate, grid_points, screen_to_target
from gebold.gaze import FineTuning, GazeModel
from gebold.inference import to_screen

SCREEN = (1920, 1080)


def model(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [16, 32, 16, 16, 8, 8, 2]
    weights = []
    for n_in, n_out in pairwise(sizes):
        weights += [rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)), np.zeros(n_out)]
    return GazeModel(weights, {"license": "CC-BY-NC-SA-4.0"})


def test_grid_points_row_by_row_with_margin():
    pts = grid_points(SCREEN, cols=3, rows=2, margin=0.1)
    np.testing.assert_allclose(pts[0], [192, 108])
    np.testing.assert_allclose(pts[2], [1728, 108])
    np.testing.assert_allclose(pts[-1], [1728, 972])
    assert len(pts) == 6


@pytest.mark.parametrize("mirror", [True, False])
def test_screen_to_target_inverts_to_screen(mirror):
    pts = grid_points(SCREEN)
    targets = screen_to_target(pts, SCREEN, mirror)
    back = np.array([to_screen(t, SCREEN, mirror) for t in targets])
    np.testing.assert_allclose(back, pts)


def test_screen_to_target_mirrors_x():
    np.testing.assert_allclose(screen_to_target([[0, 0]], SCREEN), [[1, -1]])
    np.testing.assert_allclose(screen_to_target([[0, 0]], SCREEN, mirror_x=False), [[-1, -1]])


def test_session_walks_through_points():
    session = CalibrationSession(grid_points(SCREEN, 2, 1), frames_per_point=2)
    assert session.index == 0
    assert not session.add(np.zeros(16))
    assert session.add(np.ones(16))
    np.testing.assert_allclose(session.current_point, session.points[1])
    session.add(np.zeros(16))
    session.add(np.zeros(16))
    assert session.done
    assert session.current_point is None
    with pytest.raises(RuntimeError, match="finished"):
        session.add(np.zeros(16))


def test_session_samples_pair_features_with_targets():
    session = CalibrationSession(grid_points(SCREEN, 2, 1), frames_per_point=3)
    for i in range(6):
        session.add(np.full(16, i))
    x, y = session.samples(SCREEN)
    assert x.shape == (6, 16)
    np.testing.assert_allclose(x[:, 0], range(6))
    targets = screen_to_target(session.points, SCREEN)
    np.testing.assert_allclose(y[:3], np.repeat(targets[:1], 3, axis=0))
    np.testing.assert_allclose(y[3:], np.repeat(targets[1:], 3, axis=0))


def test_session_rejects_wrong_feature_size():
    with pytest.raises(ValueError, match="16 features"):
        CalibrationSession(grid_points(SCREEN)).add(np.zeros(15))


def test_calibrate_fits_the_recorded_points_without_touching_base():
    rng = np.random.default_rng(3)
    base = model()
    session = CalibrationSession(grid_points(SCREEN), frames_per_point=4)
    centres = rng.uniform(-1, 1, (9, 16))
    while not session.done:
        session.add(centres[session.index] + rng.normal(0, 0.01, 16))
    x, y = session.samples(SCREEN)
    before = np.linalg.norm(base.predict(x) - y, axis=1).mean()
    calibrated = calibrate(base, session, SCREEN, settings=FineTuning(epochs=300, lr=3e-3))
    after = np.linalg.norm(calibrated.predict(x) - y, axis=1).mean()
    assert after < before / 3
    assert "9 points x 4 frames" in calibrated.metadata["calibration"]
    assert calibrated.metadata["license"] == "CC-BY-NC-SA-4.0"
    assert "calibration" not in base.metadata


def test_calibrate_needs_a_finished_session():
    with pytest.raises(RuntimeError, match="not finished"):
        calibrate(model(), CalibrationSession(grid_points(SCREEN)), SCREEN)
