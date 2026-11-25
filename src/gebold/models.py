"""Where the model files live and how to get them.

The thesis models (E-CLanDet, F-CLanDet and DEFC-S) are converted from the original Keras ``.h5``
files with ``gebold-export``. The SCRFD face detector is downloaded from the InsightFace releases.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ENV_MODELS_DIR: Final = "GEBOLD_MODELS_DIR"


@dataclass(frozen=True)
class ThesisModel:
    """A model of the thesis and the original ``.h5`` it is exported from.

    Attributes:
        name: base name of the exported file.
        source: path of the original model folder, relative to ``EyeCNN/models`` of the thesis
            code.
        input_shape: input shape without the batch axis.
        suffix: ``.onnx`` for the networks run with ONNX Runtime, ``.npz`` for the gaze model,
            which is evaluated and fine-tuned in numpy.
    """

    name: str
    source: str
    input_shape: tuple[int, ...]
    suffix: str = ".onnx"

    @property
    def filename(self) -> str:
        """File name of the exported model."""
        return f"{self.name}{self.suffix}"


# Final models of the thesis, checked against its tables (E-CLanDet: tables 5.7-6.4,
# F-CLanDet: tables 6.5-6.8; DEFC-S: the default model of the thesis analysis scripts).
E_CLANDET: Final = ThesisModel("e_clandet", "ibug_phd/model_28_22.05.13_00.37.02", (128, 256, 3))
F_CLANDET: Final = ThesisModel("f_clandet", "face_phd/model_3_22.05.07_18.11.35", (256, 256, 3))
DEFC_S: Final = ThesisModel(
    "defc_s", "gaze_phd/guadiana_3c/model_376_23.04.08_10.15.42", (16,), ".npz"
)
THESIS_MODELS: Final = (E_CLANDET, F_CLANDET, DEFC_S)

# Licence of the thesis weights, written into every exported file.
WEIGHTS_LICENSE: Final = "CC-BY-NC-SA-4.0"
WEIGHTS_COPYRIGHT: Final = (
    "Copyright (c) 2022-2023 Universidad Publica de Navarra (author: Andoni Larumbe Bergera)"
)
WEIGHTS_NOTICE: Final = (
    "Weights of the thesis 'Gaze Estimation Based on Machine Learning'. Licensed under CC BY-NC-SA"
    " 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only; works"
    " based on them must use the same licence."
)

SCRFD_FILENAME: Final = "det_500m.onnx"
SCRFD_URL: Final = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
)
SCRFD_ZIP_SHA256: Final = "57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72"


def models_dir() -> Path:
    """Directory of the model files: ``$GEBOLD_MODELS_DIR`` or ``./models``."""
    return Path(os.environ.get(ENV_MODELS_DIR, "models"))


def sha256(path: Path) -> str:
    """SHA-256 of a file, in hexadecimal."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Download ``url`` to ``target`` trusting the certificates of the operating system.

    Python's own certificate list does not include the certificates that some antivirus programs
    and corporate proxies add to Windows, and the download then fails with
    ``CERTIFICATE_VERIFY_FAILED``; ``truststore`` uses the system store instead.
    """
    import truststore  # noqa: PLC0415

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with urllib.request.urlopen(url, context=context) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_scrfd(directory: Path | None = None) -> Path:
    """Return the SCRFD-500MF model, downloading it from InsightFace if it is not there yet.

    The archive (``buffalo_sc.zip``, 15 MB) is checked against its SHA-256 before extracting
    ``det_500m.onnx``; the recognition model it also contains is discarded.
    """
    directory = directory or models_dir()
    target = directory / SCRFD_FILENAME
    if target.exists():
        return target
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "buffalo_sc.zip"
        download(SCRFD_URL, archive)
        if (digest := sha256(archive)) != SCRFD_ZIP_SHA256:
            raise RuntimeError(f"unexpected SHA-256 for {SCRFD_URL}: {digest}")
        with (
            zipfile.ZipFile(archive) as zf,
            zf.open(SCRFD_FILENAME) as src,
            target.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
    return target
