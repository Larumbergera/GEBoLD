"""Reproduce the landmark and gaze results of the thesis on GI4E and I2Head.

Usage::

    gebold-evaluate --gi4e GI4E_pts --i2head I2Head_pts --grid grid_points_camera.mat

``GI4E_pts`` and ``I2Head_pts`` hold the images with their ``.pts`` annotations (six eye
landmarks, 1-based as written by MATLAB). ``grid_points_camera.mat`` is the I2Head grid in camera
coordinates (millimetres). The databases are not distributed with GEBoLD.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from gebold.pipeline import GazePipeline

# F-CLanDet landmark indices of the six eye landmarks, in E-CLanDet order.
FACE_TO_EYE: Final = [1, 7, 2, 3, 8, 4]
THRESHOLDS: Final = (1.0, 2.5, 5.0)  # epmax thresholds, % of the inter-pupillary distance

# Normalisation of the point of gaze used to train DEFC (``data_preprocess.prepGaze``): the U2Eyes
# grid spans [-11, 11] cm around the camera; the I2Head grid is moved from the camera (above the
# screen) to its own centre, (0.1, 16.75) cm in camera coordinates.
GRID_HALF_CM: Final = 11.0
I2HEAD_GRID_CENTRE_CM: Final = np.array([0.1, 16.75])
# Grid point of each position index, 1-based as in ``i2head_facial_json.m``.
I2HEAD_17: Final = list(range(65, 82))
I2HEAD_65: Final = [*range(1, 65), 81]


@dataclass(frozen=True)
class Reference:
    """Result published in the thesis (MTCNN as face detector)."""

    pupils: float
    eyes: float
    accuracy: tuple[float, float, float]
    table: str


THESIS: Final = {
    ("E-CLanDet", "GI4E"): Reference(1.24, 2.69, (10.11, 97.33, 100.0), "6.3/6.4"),
    ("E-CLanDet", "I2Head"): Reference(1.11, 1.99, (25.11, 97.23, 99.96), "6.3/6.4"),
    ("F-CLanDet", "GI4E"): Reference(1.57, 2.72, (3.07, 76.21, 100.0), "6.7/6.8"),
    ("F-CLanDet", "I2Head"): Reference(1.53, 2.28, (10.63, 79.45, 99.82), "6.7/6.8"),
}
THESIS_GAZE_I2HEAD_CM: Final = (9.85, 4.03)  # DEFC-S, table 7.7


def read_pts(path: Path) -> NDArray[np.float64]:
    """Landmarks of a ``.pts`` file, converted to 0-based pixel coordinates."""
    body = path.read_text().split("{", 1)[1].split("}", 1)[0]
    return np.array(body.split(), dtype=np.float64).reshape(-1, 2) - 1


def landmark_errors(gt: NDArray, pred: NDArray) -> tuple[float, float, float]:
    """Errors of six eye landmarks in % of the inter-pupillary distance.

    Returns:
        ``(mean pupil error, mean error of the six landmarks, epmax)``, where epmax is the larger
        of the two pupil errors (Jesorsky's measure).
    """
    ipd = np.linalg.norm(gt[1] - gt[4])
    err = np.linalg.norm(gt - pred, axis=1) / ipd * 100
    return float((err[1] + err[4]) / 2), float(err.mean()), float(max(err[1], err[4]))


@dataclass
class LandmarkStats:
    """Summary of landmark errors over a database."""

    n: int
    pupils: tuple[float, float]
    eyes: tuple[float, float]
    accuracy: tuple[float, ...]

    @classmethod
    def from_rows(cls, rows: list[tuple[float, float, float]]) -> LandmarkStats:
        """Build from the output of :func:`landmark_errors` for each image."""
        a = np.array(rows, dtype=np.float64).reshape(-1, 3)
        return cls(
            n=len(a),
            pupils=(float(a[:, 0].mean()), float(a[:, 0].std())),
            eyes=(float(a[:, 1].mean()), float(a[:, 1].std())),
            accuracy=tuple(float(100 * (a[:, 2] <= t).mean()) for t in THRESHOLDS),
        )


def load_i2head_grid(path: Path) -> NDArray[np.float64]:
    """I2Head grid points in camera coordinates (mm), from ``grid_points_camera.mat``."""
    from scipy.io import loadmat  # noqa: PLC0415 (optional, 'eval' extra)

    return np.asarray(loadmat(str(path))["grid_points_camera"], dtype=np.float64)


def i2head_target(filename: str, grid: NDArray) -> NDArray[np.float64]:
    """Normalised point of gaze of an I2Head image, as used to train and test DEFC."""
    m = re.match(r"user_\d+_(17|65)_points_.+_(\d+)\.png$", filename)
    if m is None:
        raise ValueError(f"not an I2Head file name: {filename}")
    point = (I2HEAD_17 if m[1] == "17" else I2HEAD_65)[int(m[2]) - 1]
    look_at_cm = grid[point - 1, :2] / 10
    return (look_at_cm - I2HEAD_GRID_CENTRE_CM) / GRID_HALF_CM


@dataclass
class DatasetResult:
    """Evaluation of one database."""

    name: str
    n_images: int
    e_clandet: LandmarkStats
    f_clandet: LandmarkStats
    missed: list[str] = field(default_factory=list)
    gaze_cm: dict[str, tuple[float, float]] | None = None


def evaluate_dataset(
    pipeline: GazePipeline,
    folder: Path,
    grid: NDArray | None = None,
    limit: int | None = None,
) -> DatasetResult:
    """Run the pipeline on every image of ``folder``; with ``grid``, also score the gaze."""
    files = sorted(folder.glob("*.png"))[:limit]
    e_rows, f_rows, gaze, missed = [], [], [], []
    for png in files:
        result = pipeline.process(cv2.imread(str(png)))
        if result is None:
            missed.append(png.name)
            continue
        gt = read_pts(png.with_suffix(".pts"))
        e_rows.append(landmark_errors(gt, result.eye_landmarks))
        f_rows.append(landmark_errors(gt, result.face_landmarks[FACE_TO_EYE]))
        if grid is not None:
            gaze.append(np.abs(result.pog - i2head_target(png.name, grid)) * GRID_HALF_CM)
    return DatasetResult(
        folder.name.removesuffix("_pts"),
        len(files),
        LandmarkStats.from_rows(e_rows),
        LandmarkStats.from_rows(f_rows),
        missed,
        gaze_summary(gaze) if gaze else None,
    )


def gaze_summary(abs_errors_cm: list[NDArray]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of the Euclidean, x and y gaze errors (cm)."""
    g = np.array(abs_errors_cm)
    parts = {"d": np.linalg.norm(g, axis=1), "dx": g[:, 0], "dy": g[:, 1]}
    return {k: (float(v.mean()), float(v.std())) for k, v in parts.items()}


def _landmark_line(model: str, db: str, s: LandmarkStats) -> str:
    acc = " / ".join(f"{a:.2f}" for a in s.accuracy)
    line = f"  {model:9s} {s.pupils[0]:5.2f} {s.eyes[0]:5.2f}  {acc}"
    ref = THESIS.get((model, db))
    if ref:
        ref_acc = " / ".join(f"{a:.2f}" for a in ref.accuracy)
        line += f"   (thesis {ref.table}: {ref.pupils:.2f} {ref.eyes:.2f}  {ref_acc})"
    return line


def format_report(results: list[DatasetResult]) -> str:
    """Human-readable comparison with the thesis tables."""
    lines = []
    for r in results:
        found = r.n_images - len(r.missed)
        lines.append(f"{r.name}: {r.n_images} images, face found in {found}")
        lines.append("  model     pupil  eyes  epmax<=1 / 2.5 / 5 (% IPD)")
        lines.append(_landmark_line("E-CLanDet", r.name, r.e_clandet))
        lines.append(_landmark_line("F-CLanDet", r.name, r.f_clandet))
        if r.gaze_cm:
            (d, sd), (dx, _), (dy, _) = r.gaze_cm["d"], r.gaze_cm["dx"], r.gaze_cm["dy"]
            ref_d, ref_sd = THESIS_GAZE_I2HEAD_CM
            lines.append(
                f"  DEFC-S gaze error {d:.2f} +- {sd:.2f} cm (dx {dx:.2f}, dy {dy:.2f})"
                f"   (thesis 7.7: {ref_d:.2f} +- {ref_sd:.2f})"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``gebold-evaluate``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gi4e", type=Path, help="GI4E_pts folder")
    parser.add_argument("--i2head", type=Path, help="I2Head_pts folder")
    parser.add_argument("--grid", type=Path, help="I2Head grid_points_camera.mat (gaze error)")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="images per database")
    parser.add_argument("--json", type=Path, default=None, help="also write the results here")
    args = parser.parse_args(argv)
    if not (args.gi4e or args.i2head):
        parser.error("give --gi4e and/or --i2head")
    pipeline = GazePipeline(args.models_dir)
    results = []
    if args.gi4e:
        results.append(evaluate_dataset(pipeline, args.gi4e, limit=args.limit))
    if args.i2head:
        grid = load_i2head_grid(args.grid) if args.grid else None
        results.append(evaluate_dataset(pipeline, args.i2head, grid, args.limit))
    print(format_report(results))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
