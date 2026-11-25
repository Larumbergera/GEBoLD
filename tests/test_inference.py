import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from gebold.geometry import BBox
from gebold.inference import LandmarkDetector, normalise_image, to_screen

LANDMARKS = np.array([[10, 20], [100, 60], [250, 120]], np.float32)  # in crop pixels (256x128)


def save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture
def fake_landmark_model(tmp_path):
    """Outputs LANDMARKS plus the mean of the input image, so normalisation can be checked."""
    const = numpy_helper.from_array(LANDMARKS.reshape(1, -1), "const")
    axes = numpy_helper.from_array(np.array([1], np.int64), "axes")
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["mean", "axes"], ["mean2d"]),
        helper.make_node("Add", ["mean2d", "const"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fake_clandet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 128, 256, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", LANDMARKS.size])],
        [const, axes],
    )
    return save(graph, tmp_path / "fake_clandet.onnx")


def test_normalise_image_range():
    img = np.array([[[0, 127.5, 255]]]).astype(np.uint8)
    np.testing.assert_allclose(normalise_image(img), [[[-1, -0.003921568, 1]]], atol=1e-6)


def test_landmark_detector_reads_model_shapes(fake_landmark_model):
    det = LandmarkDetector(fake_landmark_model)
    assert det.input_size == (128, 256)
    assert det.n_landmarks == 3


@pytest.mark.parametrize(("value", "offset"), [(255, 1.0), (0, -1.0)])
def test_landmark_detector_normalises_and_maps_back(fake_landmark_model, value, offset):
    frame = np.full((480, 640, 3), value, np.uint8)
    box = BBox(100, 50, 128, 64)  # half the model input in both axes
    got = LandmarkDetector(fake_landmark_model).predict(frame, box)
    expected = (LANDMARKS + offset) * [128 / 256, 64 / 128] + [100, 50]
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_landmark_detector_rejects_empty_box(fake_landmark_model):
    with pytest.raises(ValueError, match="empty"):
        LandmarkDetector(fake_landmark_model).predict(
            np.zeros((10, 10, 3), np.uint8), BBox(0, 0, 0, 5)
        )


def test_to_screen_mirrors_x_by_default():
    np.testing.assert_allclose(to_screen([0.5, -0.5], (1000, 800)), [250, 200])
    np.testing.assert_allclose(to_screen([0.5, -0.5], (1000, 800), mirror_x=False), [750, 200])
    np.testing.assert_allclose(to_screen([0, 0], (1000, 800)), [500, 400])
