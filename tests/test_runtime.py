import pytest

from gebold import runtime


def test_unknown_device(tmp_path):
    with pytest.raises(ValueError, match="unknown device"):
        runtime.session(tmp_path / "x.onnx", "tpu")


def test_cuda_without_gpu_build(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        runtime.session(tmp_path / "x.onnx", "cuda")
