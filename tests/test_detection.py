import numpy as np
import pytest

from gebold.detection import STRIDES, anchor_centres, decode, letterbox, nms
from gebold.geometry import BBox

SIZE = (64, 64)  # (width, height): feature maps of 8x8, 4x4 and 2x2 cells


def empty_outputs(batched=False):
    outs = []
    for channels in (1, 4, 10):
        for stride in STRIDES:
            n = (SIZE[1] // stride) * (SIZE[0] // stride) * 2
            outs.append(np.zeros((n, channels), np.float32))
    return [o[None] for o in outs] if batched else outs


def test_anchor_centres_two_per_cell_in_row_major_order():
    c = anchor_centres(2, 3, 16)
    assert c.shape == (12, 2)
    np.testing.assert_array_equal(c[:4], [[0, 0], [0, 0], [16, 0], [16, 0]])
    np.testing.assert_array_equal(c[-1], [32, 16])


@pytest.mark.parametrize("batched", [False, True])
def test_decode_one_positive_anchor(batched):
    outs = empty_outputs()
    # stride 16, cell (row 1, col 2), second anchor -> index (1 * 4 + 2) * 2 + 1 = 13
    i = 13
    outs[1][i, 0] = 0.9
    outs[4][i] = [1, 0.5, 2, 1]  # distances in stride units: left, top, right, bottom
    outs[7][i] = np.arange(10) / 10
    boxes, scores, kps = decode([o[None] for o in outs] if batched else outs, SIZE, 0.5)
    centre = np.array([32, 16])
    np.testing.assert_allclose(boxes, [[16, 8, 64, 32]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_allclose(kps[0], centre + (np.arange(10) / 10 * 16).reshape(5, 2))
    assert centre.tolist() == [2 * 16, 1 * 16]


def test_decode_below_threshold_is_empty():
    outs = empty_outputs()
    outs[0][3, 0] = 0.4
    boxes, scores, kps = decode(outs, SIZE, 0.5)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert kps.shape == (0, 5, 2)


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.6, 0.9, 0.7])
    assert nms(boxes, scores, 0.4) == [1, 2]


def test_letterbox_keeps_aspect_ratio():
    img = np.full((480, 640, 3), 200, np.uint8)
    canvas, scale = letterbox(img, (320, 320))
    assert scale == 0.5
    assert canvas.shape == (320, 320, 3)
    assert canvas[:240].min() == 200
    assert canvas[240:].max() == 0


def test_bbox_from_detector_is_clipped():
    assert BBox(-5, 10, 30, 30).clip(20, 20) == BBox(0, 10, 20, 10)
