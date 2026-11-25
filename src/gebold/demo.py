"""Webcam demo: calibrate by looking at points, then see where you look on the screen.

Usage::

    gebold-demo [--camera 0] [--screen 0] [--device cpu|cuda] [--load calibration.npz]

Keys: ``C`` starts (or repeats) the calibration, ``Space`` or ``P`` records the point on screen,
``R`` goes back to the uncalibrated model, ``S`` saves the calibration, ``M`` flips the x axis
(for webcams that mirror the image) and ``Q`` or ``Esc`` quits. As in the original demo, each
calibration starts from the current model, so calibrating twice refines the first calibration.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from gebold.calibration import CalibrationSession, calibrate, grid_points
from gebold.gaze import GazeModel
from gebold.inference import to_screen
from gebold.models import ensure_scrfd
from gebold.pipeline import FrameResult, GazePipeline

WINDOW = "GEBoLD"
BACKGROUND = (245, 245, 245)
GAZE_COLOUR = (200, 90, 20)
TARGET_COLOUR = (40, 170, 40)
TEXT_COLOUR = (60, 60, 60)
PREVIEW_SCALE = 0.25


class Mode(Enum):
    """What the demo is doing."""

    TRACKING = auto()  # showing the point of gaze
    WAITING = auto()  # showing a calibration point, waiting for the user to record it
    RECORDING = auto()  # recording frames for the calibration point
    FITTING = auto()  # all points recorded, fine-tuning pending


@dataclass
class Options:
    """Demo settings."""

    frames_per_point: int = 5
    grid: tuple[int, int] = (3, 3)  # columns, rows
    smoothing: float = 0.3  # weight of the new estimate in the moving average
    mirror_x: bool = True
    save_path: Path = Path("calibration.npz")


class DemoController:
    """State of the demo, independent of the camera and the window."""

    def __init__(self, model: GazeModel, screen_size: tuple[int, int], options: Options):
        self.base_model = model
        self.model = model
        self.screen_size = screen_size
        self.options = options
        self.mode = Mode.TRACKING
        self.session: CalibrationSession | None = None
        self.gaze: NDArray[np.float64] | None = None
        self.message = "C: calibrate   Q: quit"

    def start_calibration(self) -> None:
        """Show the first calibration point."""
        cols, rows = self.options.grid
        points = grid_points(self.screen_size, cols, rows)
        self.session = CalibrationSession(points, self.options.frames_per_point)
        self.mode = Mode.WAITING
        self.message = "Look at the green point and press Space"

    def on_key(self, key: int) -> bool:
        """Handle a key code from ``cv2.waitKey``; returns ``False`` to quit."""
        char = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or char == "q":
            return False
        if char == "c":
            self.start_calibration()
        elif char in {" ", "p"} and self.mode is Mode.WAITING:
            self.mode = Mode.RECORDING
        elif char == "r":
            self.model, self.mode = self.base_model, Mode.TRACKING
            self.message = "Back to the uncalibrated model"
        elif char == "s":
            self.model.save(self.options.save_path)
            self.message = f"Calibration saved to {self.options.save_path}"
        elif char == "m":
            self.options.mirror_x = not self.options.mirror_x
            self.message = f"Mirror x: {'on' if self.options.mirror_x else 'off'}"
        return True

    def on_frame(self, result: FrameResult | None) -> None:
        """Update the point of gaze or record a calibration sample."""
        if result is None:
            return
        if self.mode is Mode.RECORDING and self.session is not None:
            if self.session.add(result.features):
                self.mode = Mode.FITTING if self.session.done else Mode.WAITING
            return
        if self.mode is Mode.TRACKING:
            pog = self.model.predict(result.features)
            point = to_screen(pog, self.screen_size, self.options.mirror_x)
            a = self.options.smoothing
            self.gaze = point if self.gaze is None else a * point + (1 - a) * self.gaze

    def fit(self) -> float:
        """Fine-tune on the recorded points; returns the seconds it took."""
        if self.session is None or not self.session.done:
            raise RuntimeError("no finished calibration to fit")
        start = time.perf_counter()
        self.model = calibrate(self.model, self.session, self.screen_size, self.options.mirror_x)
        elapsed = time.perf_counter() - start
        self.mode, self.session, self.gaze = Mode.TRACKING, None, None
        self.message = f"Calibrated in {elapsed:.1f} s. C: again  R: reset  S: save  Q: quit"
        return elapsed


def draw_preview(canvas: NDArray[np.uint8], frame: NDArray, result: FrameResult | None) -> None:
    """Webcam image with the detections, mirrored, in the bottom-right corner of ``canvas``."""
    view = frame.copy()
    if result is not None:
        b = result.face.box
        cv2.rectangle(view, (b.x, b.y), (b.x + b.w, b.y + b.h), TARGET_COLOUR, 2)
        for x, y in np.vstack([result.face_landmarks, result.eye_landmarks]):
            cv2.circle(view, (int(x), int(y)), 3, (0, 0, 255), -1)
    h = int(canvas.shape[0] * PREVIEW_SCALE)
    w = int(view.shape[1] * h / view.shape[0])
    view = cv2.flip(cv2.resize(view, (w, h)), 1)
    canvas[-h:, -w:] = view


def render(
    controller: DemoController, frame: NDArray, result: FrameResult | None, fps: float
) -> NDArray[np.uint8]:
    """Full-screen image for the current state."""
    width, height = controller.screen_size
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    radius = max(6, width // 120)
    session = controller.session
    if controller.mode in {Mode.WAITING, Mode.RECORDING} and session is not None:
        x, y = session.current_point.astype(int)
        filled = -1 if controller.mode is Mode.RECORDING else 3
        cv2.circle(canvas, (x, y), radius * 2, TARGET_COLOUR, filled)
        cv2.circle(canvas, (x, y), 2, (0, 0, 0), -1)
        progress = f"Point {session.index + 1}/{len(session.points)}"
        cv2.putText(canvas, progress, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOUR, 2)
    elif controller.mode is Mode.FITTING:
        text = "Calibrating..."
        cv2.putText(canvas, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOUR, 2)
    elif controller.gaze is not None:
        x, y = np.clip(controller.gaze, 0, [width - 1, height - 1]).astype(int)
        cv2.circle(canvas, (x, y), radius, GAZE_COLOUR, -1)
    status = f"{controller.message}   {fps:.0f} fps"
    if result is None:
        status += "   (no face)"
    cv2.putText(canvas, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOUR, 2)
    draw_preview(canvas, frame, result)
    return canvas


def screen_geometry(index: int) -> tuple[int, int, int, int]:
    """``(x, y, width, height)`` of monitor ``index``."""
    import screeninfo  # noqa: PLC0415 ('demo' extra)

    monitor = screeninfo.get_monitors()[index]
    return monitor.x, monitor.y, monitor.width, monitor.height


def open_window(x: int, y: int) -> None:
    """Full-screen window on the monitor whose top-left corner is ``(x, y)``."""
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(WINDOW, x, y)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def run(args: argparse.Namespace) -> int:
    """Main loop."""
    ensure_scrfd(args.models_dir)  # downloads the face detector the first time
    pipeline = GazePipeline(args.models_dir, device=args.device)
    model = GazeModel.load(args.load) if args.load else pipeline.gaze_model
    x0, y0, width, height = screen_geometry(args.screen)
    options = Options(args.frames, (args.cols, args.rows), args.smoothing, not args.no_mirror)
    options.save_path = args.save
    controller = DemoController(model, (width, height), options)
    source = args.video if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"cannot open video source {source!r}", file=sys.stderr)
        return 1
    open_window(x0, y0)
    fps, tick = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process(frame)
            controller.on_frame(result)
            now = time.perf_counter()
            fps, tick = 0.9 * fps + 0.1 / max(now - tick, 1e-6), now
            cv2.imshow(WINDOW, render(controller, frame, result, fps))
            if controller.mode is Mode.FITTING:
                cv2.waitKey(1)  # let the "Calibrating..." screen show
                controller.fit()
            if not controller.on_key(cv2.waitKey(1)):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--video", type=str, default=None, help="video file instead of webcam")
    parser.add_argument("--screen", type=int, default=0, help="monitor index")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--load", type=Path, default=None, help="calibration saved before")
    parser.add_argument("--save", type=Path, default=Path("calibration.npz"))
    parser.add_argument("--cols", type=int, default=3, help="calibration points per row")
    parser.add_argument("--rows", type=int, default=3, help="calibration points per column")
    parser.add_argument("--frames", type=int, default=5, help="frames recorded per point")
    parser.add_argument("--smoothing", type=float, default=0.3, help="0-1, 1 = no smoothing")
    parser.add_argument("--no-mirror", action="store_true", help="webcam already mirrors x")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``gebold-demo``."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
