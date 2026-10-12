# Transformer-Based Isolated Sign Language Recognition

A semester project exploring **isolated American Sign Language (ASL) word recognition** from short video clips. The proposed pipeline extracts hand and body landmarks from video, then classifies their temporal sequence. An LSTM will serve as a baseline, followed by a Transformer encoder.

> **Status:** A 100-video Hugging Face inspection sample is downloaded. Landmark extraction and model training have not yet been run. Results must be added only after running and recording experiments.

## Project goals

- Prepare a documented, reproducible subset of WLASL with its official labels and splits.
- Extract fixed-length sequences of hand and pose landmarks from each clip.
- Train and compare an LSTM baseline and a Transformer encoder on identical data splits.
- Evaluate performance with Top-1/Top-5 accuracy, macro precision/recall/F1, and a confusion matrix.
- Record data coverage, preprocessing choices, configuration, and actual results.

## Dataset recommendation

For the initial setup, use the [Voxel51/WLASL Hugging Face dataset](https://huggingface.co/datasets/Voxel51/WLASL), which its dataset card describes as a FiftyOne dataset with 11,980 samples. Use the project downloader below to load a bounded sample into the local data directory. The Hub copy is a third-party distribution, not guaranteed to be WLASL100 or to match the official split protocol.

Before downloading or using data, read the dataset card's [C-UDA terms](https://github.com/microsoft/Computational-Use-of-Data-Agreement/releases/tag/v1.0) and ensure your intended use complies. For benchmark claims, verify the Hub copy's labels and split fields against the [official WLASL repository](https://github.com/dxli94/WLASL).

**In short:** Hugging Face/FiftyOne is the selected path for project setup and initial exploration; the official WLASL repository remains the reference for provenance and benchmark protocol. Avoid treating a hosted copy or Kaggle upload as the official benchmark without checking its provenance, version, labels, splits, and terms.

See [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) for the acquisition checklist, end-to-end workflow, proposed folder layout, and experiment plan.

## Proposed approach

1. Audit annotations, split definitions, class counts, and available videos.
2. Verify the Hub sample against WLASL metadata and obtain official train/validation/test assignments before benchmark training.
3. Sample/pad each clip to an initial target of 32 frames and extract MediaPipe landmarks (pipeline is implemented; extraction still needs to be run).
4. Compare an LSTM baseline and Transformer with the same features and splits.
5. Evaluate on the untouched test split and report measured results.

The initial landmark representation is 21 left-hand + 21 right-hand + 33 pose landmarks, with x/y/z coordinates: 75 × 3 = **225 values per frame**. Fix and document landmark ordering, coordinate normalization, missing-landmark handling, and sequence masks before training.

## Project layout

The repository contains a Hugging Face downloader and MediaPipe landmark extraction pipeline. Model, training, and evaluation code are still planned; see [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) for the implemented structure and next steps.

## Environment and data handling

### Run the tests

The existing `.venv` was created with Python 3.14, which this project does not support. Keep it as-is and create a separate Python 3.11 environment in the integrated terminal at the project root:

```sh
py install 3.11
py -3.11 -m venv .venv311
```

In VS Code, open the Command Palette, choose **Python: Select Interpreter**, and select `.venv311\Scripts\python.exe`. Close the old terminal (especially one showing `(.venv)`) and open a new integrated terminal. Verify the selected interpreter before installing anything:

```sh
python --version
python -c "import sys; print(sys.executable)"
```

The version must be **Python 3.11.x**, and the printed executable must point inside `.venv311`. If either check still shows Python 3.14 or `.venv`, stop: select `.venv311` in the interpreter picker and open another fresh terminal. Do not run pip or the extractor until these checks pass. Then install dependencies and run the tests:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The last command runs the automated tests in `tests/`. They do **not** download the dataset or require the videos.

The `Package ... requires a different Python` message means pip is still running under Python 3.14. The `Landmark manifest not found` message is expected until extraction successfully completes; it is not a separate download problem. Use the same `.venv311` interpreter for dependency installation and extraction.

### Download data and extract landmarks

Run commands from the same integrated terminal. To extract landmarks from three downloaded videos:

```sh
python scripts/extract_landmarks.py --limit 3
```

To process every row in the downloaded manifest, omit `--limit 3`. To download data first, use the downloader; it defaults to 100 samples:

```sh
python scripts/download_wlasl_hf.py
```

You can change the number or explicitly request the full Hub dataset:

```sh
python scripts/download_wlasl_hf.py --max-samples 250
python scripts/download_wlasl_hf.py --all
```

Videos are copied under `data/raw/huggingface_wlasl/videos/`; the generated metadata manifest is saved under `data/processed/manifests/`. The full dataset is listed as 11,980 samples, so check available disk space and the data terms before using `--all`. Downloaded data and generated manifests are ignored by Git.

The existing 100-video download contains 19 glosses and should only be used to validate downloading and preprocessing. It is not a balanced WLASL100 sample or an official train/validation/test split.

After extraction, check that the arrays were actually saved and have the expected shapes:

```sh
python scripts/check_landmarks.py
```

The checker reads `data/processed/manifests/landmark_manifest.csv` and validates each listed `.npz` file. Look for `All listed feature files passed validation.` and check that rows are marked `processed` or `already_exists`, not `failed`. It prints each gloss, feature shape (normally `(32, 75, 3)`), and number of detected landmarks. If there are validation issues, it prints which file needs attention and exits with an error. Feature arrays are under `data/processed/landmarks/`. If a video's metadata claims exactly one more frame than OpenCV can decode, extraction repeats the final successfully decoded frame for the last time step; other decoding failures remain errors.

If MediaPipe reports it is missing, check `python --version` first. It must show 3.11.x; then rerun `python -m pip install -r requirements-dev.txt` in the selected `.venv311` terminal. `pyproject.toml` defines the package and runtime dependencies; training dependencies will be added when model implementation begins.


## Results

No experiments have been run yet. Replace the placeholders only with measured results and include the dataset subset and split in each row.

| Model | Dataset/split | Classes | Top-1 | Top-5 | Macro F1 |
|---|---|---:|---:|---:|---:|
| LSTM baseline | Not run | — | — | — | — |
| Transformer | Not run | — | — | — | — |

## References

- Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison.* WACV, 1459–1469. [Paper and official project](https://github.com/dxli94/WLASL)
- [Official WLASL repository and data agreement](https://github.com/dxli94/WLASL)
- [Voxel51 WLASL on Hugging Face](https://huggingface.co/datasets/Voxel51/WLASL)

## Team

- Kushagra Duggar
- Nikhilesh Joshi

Affiliation: Department of Information Technology, National Institute of Technology Karnataka, Surathkal.
