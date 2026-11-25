# Run `just` (or `uv run just`) to list the recipes.

set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Quieter TensorFlow when exporting the models
export TF_CPP_MIN_LOG_LEVEL := "3"

# List recipes
default:
    @just --list

# Install the dev environment (add `--extra demo` for the webcam demo)
sync *args:
    uv sync {{args}}

# Format the code in place
fmt:
    uv run ruff format src tests
    uv run ruff check --select I --fix src tests

# Check formatting without changing files
format-check:
    uv run ruff format --check src tests

# Lint (pycodestyle, pyflakes, bugbear, pylint, docstrings...)
lint:
    uv run ruff check src tests

# Cyclomatic complexity and size limits only (also part of `lint`)
complexity:
    uv run ruff check --select C90,PLR0911,PLR0912,PLR0913,PLR0915 src tests

# Run the tests
test *args:
    uv run pytest {{args}}

# Run the tests with coverage
cov:
    uv run pytest --cov=gebold --cov-report=term-missing

# Everything CI runs
check: format-check lint cov

# Folder with the thesis Keras models (EyeCNN/models of the original code)
thesis_models := env("GEBOLD_THESIS_MODELS", "../EyeCNN/models")

# Download SCRFD and export the thesis models to ONNX (into $GEBOLD_MODELS_DIR or ./models)
models:
    uv run python -c "from gebold.models import ensure_scrfd; print(ensure_scrfd())"
    uv run --extra export gebold-export --check --source-dir {{thesis_models}}

# Tests that need the model files
test-models:
    uv run pytest -m models -rs

# Databases for `just evaluate` (not distributed with GEBoLD)
gi4e := env("GEBOLD_GI4E", "../GI4E_pts")
i2head := env("GEBOLD_I2HEAD", "../I2Head_pts")
i2head_grid := env("GEBOLD_I2HEAD_GRID", "../Gaze/Geometric/grid_points_camera.mat")

# Score the pipeline on GI4E and I2Head against the thesis tables
evaluate *args:
    uv run --extra eval gebold-evaluate --gi4e {{gi4e}} --i2head {{i2head}} --grid {{i2head_grid}} {{args}}

# Webcam demo: calibrate and see where you look (e.g. `just demo --device cuda`)
demo *args:
    uv run --extra demo gebold-demo {{args}}
