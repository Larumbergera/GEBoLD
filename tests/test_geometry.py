import numpy as np
import pytest

from gebold.geometry import BBox, eyes_bbox, eyes_bbox_from_eye_centres, face_bbox


def test_clip_shrinks_box_leaving_on_the_left_and_top():
    # The original code moved the corner to 0 but kept the full width and height.
    assert BBox(-10, -20, 100, 100).clip(640, 480) == BBox(0, 0, 90, 80)


def test_clip_shrinks_box_leaving_on_the_right_and_bottom():
    assert BBox(600, 450, 100, 100).clip(640, 480) == BBox(600, 450, 40, 30)


def test_clip_box_outside_is_empty():
    assert BBox(700, 10, 50, 50).clip(640, 480).is_empty


def test_face_bbox_enlarges_by_factor():
    pts = np.array([[100, 100], [200, 100], [150, 220]])
    assert face_bbox(pts, 640, 480, factor=0.25) == BBox(87, 85, 125, 150)


def test_eyes_bbox_is_at_least_half_as_tall_as_wide():
    box = eyes_bbox([100, 200], [300, 200], 640, 480)
    assert box == BBox(100, 150, 200, 100)


def test_eyes_bbox_keeps_tall_boxes():
    box = eyes_bbox([100, 100], [200, 220], 640, 480)
    assert (box.w, box.h) == (100, 120)


def original_createbbox(img_w, img_h, left_eye, right_eye):
    """Transcription of ``createbbox`` in ``mtcnn_databases.py`` (thesis code)."""
    rate = 7 / 5
    xr, yr = left_eye
    xl, yl = right_eye
    w, h = rate * abs(xl - xr), rate * abs(yl - yr)
    if xr <= xl:
        x0, xf = xl - w, xr + w
    else:
        x0, xf = xr - w, xl + w
    y = yr if yr <= yl else yl
    h_aux = 0.5 * w
    if h_aux > h:
        y -= (h_aux - h) / 2
        h = h_aux
    dh = h * 0.1
    y0 = y - dh / 2
    yf = y0 + h + dh
    bbox = [x0, y0, xf - x0, h + dh]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    if xf > img_w:
        bbox[2] = img_w - x0
    if yf > img_h:
        bbox[3] = img_h - y0
    return list(map(int, bbox))


@pytest.mark.parametrize("seed", range(10))
def test_eyes_bbox_from_eye_centres_matches_original_inside_image(seed):
    rng = np.random.default_rng(seed)
    left = rng.uniform([250, 200], [290, 240])
    right = left + rng.uniform([60, -15], [110, 15])
    box = eyes_bbox_from_eye_centres(left, right, 640, 480)
    assert [box.x, box.y, box.w, box.h] == original_createbbox(640, 480, left, right)


def test_eyes_bbox_from_eye_centres_spans_both_eyes():
    box = eyes_bbox_from_eye_centres([300, 200], [400, 200], 640, 480)
    assert (box.x, box.w) == (260, 180)
    assert box.y < 200 < box.y + box.h


@pytest.mark.parametrize("out_size", [(128, 256), (256, 256)])
def test_crop_coordinates_round_trip(out_size):
    box = BBox(40, 30, 200, 120)
    pts = np.array([[50.0, 35.0], [239.0, 149.0], [100.5, 60.25]])
    np.testing.assert_allclose(box.from_crop(box.to_crop(pts, out_size), out_size), pts)


def test_crop_returns_box_region():
    img = np.arange(100).reshape(10, 10)
    np.testing.assert_array_equal(BBox(2, 3, 4, 2).crop(img), img[3:5, 2:6])
