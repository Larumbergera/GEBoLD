from pathlib import Path

import numpy as np
import pytest

from gebold.evaluate import (
    I2HEAD_GRID_CENTRE_CM,
    DatasetResult,
    LandmarkStats,
    evaluate_dataset,
    format_report,
    gaze_summary,
    i2head_target,
    landmark_errors,
    read_pts,
)
from gebold.models import SCRFD_FILENAME, THESIS_MODELS, models_dir
from gebold.pipeline import GazePipeline

GT = np.array([[0, 0], [10, 0], [20, 0], [80, 0], [90, 0], [100, 0]], float)  # IPD = 80


def test_read_pts_converts_to_zero_based(tmp_path):
    pts = tmp_path / "a.pts"
    pts.write_text("version: 1\nn_points: 2\n{\n1.5 2.0\n11.0 21.0\n}\n")
    np.testing.assert_allclose(read_pts(pts), [[0.5, 1.0], [10.0, 20.0]])


def test_landmark_errors_in_percent_of_ipd():
    pred = GT.copy()
    pred[1] += [0.8, 0]  # right pupil 1 % of the IPD away
    pred[4] += [0, 2.4]  # left pupil 3 %
    pupils, eyes, epmax = landmark_errors(GT, pred)
    assert pupils == pytest.approx(2.0)
    assert eyes == pytest.approx(4.0 / 6)
    assert epmax == pytest.approx(3.0)


def test_landmark_stats_accuracy_thresholds():
    stats = LandmarkStats.from_rows([(0.5, 1, 0.5), (2, 2, 2.0), (4, 4, 4.0), (9, 9, 9.0)])
    assert stats.n == 4
    assert stats.accuracy == (25.0, 50.0, 75.0)
    assert stats.pupils[0] == pytest.approx(3.875)


def test_i2head_target_moves_origin_to_grid_centre():
    grid = np.zeros((81, 3))
    grid[64] = [*(I2HEAD_GRID_CENTRE_CM * 10), 0]  # grid point 65 (1-based) at the centre
    grid[80] = [-110 + 1, 167.5 + 110, 0]  # grid point 81
    np.testing.assert_allclose(i2head_target("user_01_17_points_free_01.png", grid), [0, 0])
    np.testing.assert_allclose(
        i2head_target("user_03_65_points_Static_65.png", grid), [-1, 1], atol=1e-12
    )
    np.testing.assert_allclose(
        i2head_target("user_03_17_points_Trans_Bwd_17.png", grid), [-1, 1], atol=1e-12
    )


def test_i2head_target_rejects_other_names():
    with pytest.raises(ValueError, match="I2Head"):
        i2head_target("001_01.png", np.zeros((81, 3)))


def test_gaze_summary():
    s = gaze_summary([np.array([3.0, 4.0]), np.array([0.0, 0.0])])
    assert s["d"] == (2.5, 2.5)
    assert s["dx"] == (1.5, 1.5)


def test_report_compares_with_thesis():
    stats = LandmarkStats(10, (1.2, 0.1), (2.7, 0.2), (10.0, 97.0, 100.0))
    result = DatasetResult("GI4E", 11, stats, stats, ["x.png"], None)
    report = format_report([result])
    assert "face found in 10" in report
    assert "thesis 6.3/6.4: 1.24 2.69" in report


# --- With the real models and the original sample frames of the repo -----------------------

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "originals"


@pytest.mark.models
@pytest.mark.parametrize("database", ["GI4E", "I2Head"])
def test_real_pipeline_on_sample_frames(database):
    needed = [SCRFD_FILENAME, *(m.filename for m in THESIS_MODELS)]
    if not all((models_dir() / name).exists() for name in needed):
        pytest.skip("run `just models` first")
    result = evaluate_dataset(GazePipeline(), SAMPLES / database)
    assert result.n_images == 3
    assert result.missed == []
    # In the thesis tables the mean pupil error is 1.1-1.6 % of the IPD and every pupil is
    # within 5 %.
    assert result.e_clandet.pupils[0] < 2.0
    assert result.e_clandet.accuracy[2] == 100.0
    assert result.f_clandet.pupils[0] < 3.0


def test_pipeline_lists_missing_models(tmp_path):
    (tmp_path / SCRFD_FILENAME).write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"f_clandet\.onnx, e_clandet\.onnx, defc_s\.npz"):
        GazePipeline(tmp_path)
