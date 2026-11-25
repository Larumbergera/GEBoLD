"""Export the thesis Keras models (``.h5``): the landmark networks to ONNX, DEFC to ``.npz``.

Usage::

    gebold-export --source-dir ../EyeCNN/models [--output-dir models] [--check] [NAME ...]

Each model is converted in its own process: converting the two ResNet50 in one process needs
more memory than a 6 GB WSL machine has. tf2onnx's ONNX optimisers are disabled for the same
reason (with them, converting F-CLanDet goes past 6 GB); ONNX Runtime optimises the graph when it
loads it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from gebold import __version__
from gebold.gaze import GazeModel
from gebold.models import (
    THESIS_MODELS,
    WEIGHTS_COPYRIGHT,
    WEIGHTS_LICENSE,
    WEIGHTS_NOTICE,
    ThesisModel,
    models_dir,
)

BY_NAME = {m.name: m for m in THESIS_MODELS}


def weights_metadata(model: ThesisModel) -> dict[str, str]:
    """Licence, copyright and origin written into every exported file."""
    return {
        "license": WEIGHTS_LICENSE,
        "copyright": WEIGHTS_COPYRIGHT,
        "notice": WEIGHTS_NOTICE,
        "source_model": model.source,
    }


def add_metadata(path: Path, model: ThesisModel) -> None:
    """Write the licence, copyright and origin of the weights into an ONNX file."""
    import onnx  # noqa: PLC0415 (comes with the 'export' extra)

    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    for key, value in weights_metadata(model).items():
        proto.metadata_props.add(key=key, value=value)
    proto.producer_name, proto.producer_version = "gebold-export", __version__
    onnx.save(proto, str(path))


def dense_weights(keras_model) -> list[np.ndarray]:
    """Kernels and biases of the Dense layers of DEFC, checking it has the expected layout."""
    weights = []
    dense = [layer for layer in keras_model.layers if layer.weights]
    for i, layer in enumerate(dense):
        config = layer.get_config()
        expected = "linear" if i == len(dense) - 1 else "elu"
        if type(layer).__name__ != "Dense" or config["activation"] != expected:
            raise ValueError(f"unexpected layer {layer.name}: {config.get('activation')}")
        weights += layer.get_weights()
    rates = [layer.get_config().get("rate", 0) for layer in keras_model.layers]
    if any(rates):
        raise ValueError("dropout layers with a non-zero rate are not supported")
    return weights


def convert(model: ThesisModel, source_dir: Path, output_dir: Path, check: bool) -> float | None:
    """Convert one model; with ``check``, return the largest difference with Keras outputs."""
    import tensorflow as tf  # noqa: PLC0415 (heavy, optional dependency)

    keras_model = tf.keras.models.load_model(source_dir / model.source / "model.h5", compile=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / model.filename
    if model.suffix == ".npz":
        gaze = GazeModel(dense_weights(keras_model), weights_metadata(model))
        gaze.save(target)
        predict = gaze.predict
    else:
        import onnxruntime as ort  # noqa: PLC0415
        import tf2onnx  # noqa: PLC0415

        spec = (tf.TensorSpec((None, *model.input_shape), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(
            keras_model, input_signature=spec, opset=17, output_path=str(target), optimizers={}
        )
        add_metadata(target, model)
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])

        def predict(x):
            return session.run(None, {"input": x})[0]

    if not check:
        return None
    x = np.random.default_rng(0).uniform(-1, 1, (8, *model.input_shape)).astype(np.float32)
    return float(np.abs(keras_model.predict_on_batch(x) - predict(x)).max())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help=f"models: {', '.join(BY_NAME)} (default: all)")
    parser.add_argument("--source-dir", type=Path, required=True, help="EyeCNN/models folder")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: $GEBOLD_MODELS_DIR")
    parser.add_argument(
        "--check", action="store_true", help="compare the exported model with Keras"
    )
    parser.add_argument("--tolerance", type=float, default=0.05, help="max output difference")
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if unknown := set(args.names) - set(BY_NAME):
        parser.error(f"unknown models: {', '.join(sorted(unknown))}")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``gebold-export``."""
    args = parse_args(argv)
    output_dir = args.output_dir or models_dir()
    names = args.names or list(BY_NAME)
    if args.in_process:
        model = BY_NAME[names[0]]
        diff = convert(model, args.source_dir, output_dir, args.check)
        line = f"{model.name}: {output_dir / model.filename}"
        if diff is None:
            print(line)
            return 0
        print(f"{line} (max |keras - exported| = {diff:.2e})")
        return 0 if diff <= args.tolerance else 1
    status = 0
    for name in names:
        cmd = [sys.executable, "-m", "gebold.export", name, "--in-process"]
        cmd += ["--source-dir", str(args.source_dir), "--output-dir", str(output_dir)]
        cmd += ["--tolerance", str(args.tolerance)] + (["--check"] if args.check else [])
        status |= subprocess.run(cmd, check=False).returncode
    return status


if __name__ == "__main__":
    sys.exit(main())
