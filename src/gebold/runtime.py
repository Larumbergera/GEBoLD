"""ONNX Runtime sessions on CPU or GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import onnxruntime as ort

Device = Literal["cpu", "cuda"]

PROVIDERS: Final[dict[str, list]] = {
    "cpu": ["CPUExecutionProvider"],
    # TF32 is off: with it (the default on recent NVIDIA GPUs) landmarks moved up to 1.2 px and the
    # point of gaze up to 1 cm from the CPU results on I2Head, for 20 % more speed. CPU stays as
    # fallback for any operator CUDA does not support.
    "cuda": [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
}


def session(path: str | Path, device: Device = "cpu") -> ort.InferenceSession:
    """Open an ONNX model on ``device``.

    For ``cuda`` install ``onnxruntime-gpu`` instead of ``onnxruntime``.

    Raises:
        RuntimeError: if ``cuda`` is requested but ONNX Runtime has no CUDA support.
    """
    if device not in PROVIDERS:
        raise ValueError(f"unknown device {device!r}; use one of {', '.join(PROVIDERS)}")
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA is not available: install onnxruntime-gpu, not onnxruntime")
        if hasattr(ort, "preload_dlls"):  # CUDA and cuDNN installed as pip wheels
            ort.preload_dlls()
    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only: SCRFD's dynamic shapes flood the warnings
    # The pipeline runs several sessions one after another; spinning worker threads of one
    # session would steal the CPU from the next.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=PROVIDERS[device])
