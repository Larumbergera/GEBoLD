# GEBoLD
This repository contains a demo application showcasing the gaze estimation method proposed in the thesis titled "Gaze Estimation Based on Machine Learning".

## How it works

Each webcam frame goes through four steps:

1. [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) finds the face (the thesis used MTCNN; SCRFD is about 150 times faster on CPU with the same landmark accuracy).
2. F-CLanDet locates nine facial landmarks and E-CLanDet six eye landmarks (two ResNet50 run with ONNX Runtime). The eyes come from E-CLanDet, the nose and mouth from F-CLanDet.
3. The landmarks give a vector of 16 features: pupil-to-corner vectors, pupil positions, inter-eye distance and head yaw, roll and pitch.
4. DEFC, a small fully connected network trained on synthetic data (U2Eyes), maps the features to the point of gaze. A short calibration fine-tunes it for each user.

## Run the demo

The demo opens a full-screen window: press `C` and look at each green point, pressing `Space` on each one; after the last point the model is fine-tuned for you (under a second) and a dot follows your gaze.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Put the model files in `models/` (or point `GEBOLD_MODELS_DIR` to them): `e_clandet.onnx`, `f_clandet.onnx` and `defc_s.npz`. `just models` exports them from the original Keras models of the thesis; the SCRFD detector (`det_500m.onnx`) is downloaded automatically.
3. From the repository folder run:

   ```bash
   uv run --extra demo gebold-demo
   ```

| Key | Action |
|---|---|
| `C` | Start (or repeat) the calibration. Each calibration refines the current model |
| `Space` / `P` | Record the green point while looking at it |
| `R` | Go back to the uncalibrated model |
| `S` | Save the calibration (`--load calibration.npz` reuses it) |
| `M` | Flip the x axis, for webcams that deliver mirrored images |
| `Q` / `Esc` | Quit |

Options: `--camera N` and `--screen N` choose the webcam and the monitor, `--cols`, `--rows` and `--frames` change the calibration grid, and `--device cuda` runs the landmark networks on an NVIDIA GPU (needs `onnxruntime-gpu` instead of `onnxruntime`). Keep the head still after calibrating: the model is sensitive to head pose.

On an 8-thread CPU the whole pipeline runs at about 15 frames per second; on an RTX 5060 Ti, at about 50.

## Development

```bash
uv run just check      # format, lint, complexity limits and tests with coverage
uv run just models     # export the thesis models and download SCRFD
uv run just evaluate   # compare with the thesis tables (needs GI4E and I2Head)
```

## Data

The databases used in the thesis are not distributed with GEBoLD; request them from their authors:

- [GI4E](https://www.unavarra.es/gi4e/databases/gi4e?languageId=1) and [I2Head](https://www.unavarra.es/gi4e/databases/i2head?languageId=1), from the [databases of the GI4E group](https://www.unavarra.es/gi4e/databases?languageId=1) of the Public University of Navarre.
- [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild), from the Max Planck Institute for Informatics.

`just evaluate` scores the models on GI4E and I2Head and compares the results with the thesis tables. By default it looks for `../GI4E_pts`, `../I2Head_pts` and `../Gaze/Geometric/grid_points_camera.mat`; set `GEBOLD_GI4E`, `GEBOLD_I2HEAD` and `GEBOLD_I2HEAD_GRID` to use other paths.

## License

Copyright (c) 2020-2023 Universidad Pública de Navarra (original thesis code, author: Andoni Larumbe Bergera).  
Copyright (c) 2026 Andoni Larumbe Bergera (new code and changes).

GEBoLD is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0), the same license as the thesis. You may use, share and adapt it for non-commercial purposes, giving appropriate credit; works based on it must be distributed under the same license. Commercial use is not permitted. Because of the non-commercial clause, this is source-available software rather than open source in the OSI sense.

### Model weights

The trained weights of the thesis models (E-CLanDet, F-CLanDet and DEFC-S) are also released under CC BY-NC-SA 4.0: non-commercial use only, and models derived from them (for example, fine-tuned or converted versions) must be shared under the same license.

Copyright (c) 2022-2023 Universidad Pública de Navarra (author: Andoni Larumbe Bergera).

Every ONNX file produced by `gebold-export` carries this license and copyright in its metadata.

The SCRFD face detector is not part of this release. Its weights belong to [InsightFace](https://github.com/deepinsight/insightface#license), which makes them available for non-commercial research purposes only; GEBoLD downloads them from the InsightFace releases on first use.
