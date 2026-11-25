from itertools import pairwise

import numpy as np
import pytest

from gebold.demo import DemoController, Mode, Options, parse_args, render
from gebold.detection import Face
from gebold.gaze import GazeModel
from gebold.geometry import BBox
from gebold.pipeline import FrameResult

SCREEN = (640, 360)


def model():
    rng = np.random.default_rng(0)
    weights = []
    for n_in, n_out in pairwise([16, 32, 16, 16, 8, 8, 2]):
        weights += [rng.normal(0, 0.3, (n_in, n_out)), np.zeros(n_out)]
    return GazeModel(weights)


def result(features=None):
    face = Face(BBox(50, 40, 80, 100), 0.9, np.zeros((5, 2)))
    return FrameResult(
        face=face,
        eye_box=BBox(60, 60, 60, 20),
        face_landmarks=np.full((9, 2), 70.0),
        eye_landmarks=np.full((6, 2), 80.0),
        features=np.zeros(16, np.float32) if features is None else features,
        pog=np.zeros(2, np.float32),
    )


@pytest.fixture
def controller(tmp_path):
    options = Options(frames_per_point=2, grid=(2, 2), save_path=tmp_path / "cal.npz")
    return DemoController(model(), SCREEN, options)


def test_tracking_moves_gaze_inside_the_screen_range(controller):
    controller.on_frame(result())
    first = controller.gaze.copy()
    controller.on_frame(result(np.ones(16, np.float32)))
    assert controller.mode is Mode.TRACKING
    assert not np.allclose(controller.gaze, first)


def test_no_face_keeps_state(controller):
    controller.on_frame(None)
    assert controller.gaze is None


def test_full_calibration_flow(controller):
    rng = np.random.default_rng(1)
    assert controller.on_key(ord("c"))
    assert controller.mode is Mode.WAITING
    controller.on_frame(result())  # frames are ignored until the user presses Space
    assert controller.session.index == 0
    for point in range(4):
        controller.on_key(ord(" "))
        assert controller.mode is Mode.RECORDING
        for _ in range(2):
            controller.on_frame(result(rng.normal(point, 0.1, 16).astype(np.float32)))
        assert controller.mode is (Mode.WAITING if point < 3 else Mode.FITTING)
    controller.fit()
    assert controller.mode is Mode.TRACKING
    assert controller.model is not controller.base_model
    assert "calibration" in controller.model.metadata


def test_reset_save_mirror_and_quit_keys(controller):
    controller.model = model()
    assert controller.on_key(ord("r"))
    assert controller.model is controller.base_model
    assert controller.on_key(ord("s"))
    assert controller.options.save_path.exists()
    assert controller.on_key(ord("m"))
    assert controller.options.mirror_x is False
    assert controller.on_key(-1)  # no key pressed
    assert not controller.on_key(ord("q"))
    assert not controller.on_key(27)


def test_render_shapes_and_preview(controller):
    frame = np.zeros((480, 640, 3), np.uint8)
    controller.on_frame(result())
    img = render(controller, frame, result(), fps=15)
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    preview_h = int(360 * 0.25)
    assert img[-preview_h:, -10:].max() > 0  # webcam preview drawn in the corner
    controller.start_calibration()
    img = render(controller, frame, None, fps=15)
    x, y = controller.session.current_point.astype(int)
    assert tuple(img[y, x]) == (0, 0, 0)  # centre of the target


def test_parse_args_defaults():
    args = parse_args([])
    assert (args.camera, args.screen, args.device) == (0, 0, "cpu")
    assert (args.cols, args.rows, args.frames) == (3, 3, 5)
