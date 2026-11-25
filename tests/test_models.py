import hashlib
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from gebold import models
from gebold.detection import FaceDetector
from gebold.export import add_metadata, parse_args
from gebold.gaze import GazeModel
from gebold.models import (
    DEFC_S,
    E_CLANDET,
    ENV_MODELS_DIR,
    F_CLANDET,
    THESIS_MODELS,
    ensure_scrfd,
    models_dir,
)


def test_models_dir_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_MODELS_DIR, str(tmp_path))
    assert models_dir() == tmp_path
    monkeypatch.delenv(ENV_MODELS_DIR)
    assert models_dir() == Path("models")


def test_thesis_models_have_unique_names():
    assert len({m.filename for m in THESIS_MODELS}) == len(THESIS_MODELS) == 3


def test_ensure_scrfd_does_not_download_when_present(monkeypatch, tmp_path):
    (tmp_path / models.SCRFD_FILENAME).write_bytes(b"model")
    monkeypatch.setattr(models, "download", pytest.fail)
    assert ensure_scrfd(tmp_path) == tmp_path / models.SCRFD_FILENAME


def fake_download(content: bytes):
    def fake(_url, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr(models.SCRFD_FILENAME, content)
            zf.writestr("w600k_mbf.onnx", b"not needed")

    return fake


def test_ensure_scrfd_extracts_detector_after_checking_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "download", fake_download(b"detector"))
    fake = tmp_path / "probe.zip"
    fake_download(b"detector")(None, fake)
    monkeypatch.setattr(models, "SCRFD_ZIP_SHA256", hashlib.sha256(fake.read_bytes()).hexdigest())
    out = ensure_scrfd(tmp_path / "m")
    assert out.read_bytes() == b"detector"
    assert sorted(p.name for p in out.parent.iterdir()) == [models.SCRFD_FILENAME]


def test_ensure_scrfd_rejects_wrong_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "download", fake_download(b"tampered"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        ensure_scrfd(tmp_path)
    assert not (tmp_path / models.SCRFD_FILENAME).exists()


def test_add_metadata_writes_weights_licence(tmp_path):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "tiny",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "tiny.onnx"
    onnx.save(onnx.helper.make_model(graph), path)
    add_metadata(path, THESIS_MODELS[0])
    add_metadata(path, THESIS_MODELS[0])  # idempotent
    props = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert props["license"] == "CC-BY-NC-SA-4.0"
    assert "Universidad Publica de Navarra" in props["copyright"]
    assert "non-commercial" in props["notice"]
    assert props["source_model"] == THESIS_MODELS[0].source
    assert len(onnx.load(path).metadata_props) == 4


def test_export_rejects_unknown_model(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["nope", "--source-dir", str(tmp_path)])


def test_export_defaults_to_all_models(tmp_path):
    args = parse_args(["--source-dir", str(tmp_path)])
    assert args.names == []
    assert not args.check


# --- Tests with the real model files (skipped unless `just models` has been run) -------------


def model_path(name: str) -> Path:
    path = models_dir() / name
    if not path.exists():
        pytest.skip(f"{path} not found; run `just models`")
    return path


@pytest.mark.models
def test_real_scrfd_finds_no_face_in_blank_frame():
    det = FaceDetector(model_path(models.SCRFD_FILENAME))
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []


@pytest.mark.models
@pytest.mark.parametrize("model", [E_CLANDET, F_CLANDET], ids=lambda m: m.name)
def test_real_landmark_models_load_with_expected_shapes(model):
    session = ort.InferenceSession(str(model_path(model.filename)))
    assert tuple(session.get_inputs()[0].shape[1:]) == model.input_shape
    out = session.run(None, {"input": np.zeros((2, *model.input_shape), np.float32)})[0]
    assert out.shape == (2, {"e_clandet": 12, "f_clandet": 18}[model.name])
    assert session.get_modelmeta().custom_metadata_map["license"] == "CC-BY-NC-SA-4.0"


@pytest.mark.models
def test_real_gaze_model_has_defc_layout_and_licence():
    model = GazeModel.load(model_path(DEFC_S.filename))
    assert [w.shape[-1] for w in model.weights[::2]] == [32, 16, 16, 8, 8, 2]
    assert model.metadata["license"] == "CC-BY-NC-SA-4.0"
    assert model.metadata["source_model"] == DEFC_S.source
