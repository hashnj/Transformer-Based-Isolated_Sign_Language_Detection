# Implementation Guide

This guide turns the project proposal into an implementation sequence. The repository includes the Hugging Face/FiftyOne downloader and MediaPipe Holistic feature extraction. Model training and evaluation remain future implementation phases.

## 1. Define the experiment before downloading

The target task is **isolated sign classification**: one short clip contains one sign/gloss, and the model predicts that gloss. It is not continuous sign-language translation.

For initial exploration, load the **Voxel51/WLASL Hugging Face copy**. Its dataset card lists 11,980 samples and labels the license as `other`, with the C-UDA terms described below. This copy is not automatically a WLASL100 subset. For benchmark experiments, verify labels, class selection, and train/validation/test assignments against the official WLASL release. The official repository describes WLASL100 as the 100 most frequent glosses from the WLASL annotation file. Do not make a random clip-level split.

Before acquisition:

- Read the Hugging Face dataset card and the [C-UDA terms](https://github.com/microsoft/Computational-Use-of-Data-Agreement/releases/tag/v1.0); confirm your intended use is permitted.
- For benchmark results, use the [official WLASL repository](https://github.com/dxli94/WLASL) to verify annotation/split provenance.
- Confirm that the intended academic use and local storage comply with the agreement.
- Ask the supervisor whether a small, documented WLASL100 experiment is sufficient for the semester deliverable.
- Decide on a fixed random seed, initial sequence length (32 frames), and landmark representation.

## 2. Dataset acquisition: which route to use?

| Route | Best use | Advantages | Checks and limitations |
|---|---|---|---|
| Voxel51/WLASL on Hugging Face + FiftyOne | **Selected path for initial exploration** | Dataset card lists 11,980 samples; FiftyOne loader can fetch samples and metadata | Third-party copy. Check terms, labels, release, and split compatibility before benchmark use. Not guaranteed to be WLASL100. |
| Official WLASL GitHub metadata and download/request instructions | Ground truth for benchmark protocol and recovery of missing videos | Provides the annotation schema, source download instructions, data agreement, and missing/preprocessed video request information | Source URLs may be unavailable. Follow current instructions and request missing videos through the official process. |
| Kaggle upload | Only if a particular upload is independently verified | May be easy to download or already packaged | Uploads can be unofficial, stale, incomplete, relabelled, or missing official split metadata. Verify provenance and terms; do not use as the source of truth. |

**Practical recommendation:** use the supplied script for a small Hugging Face sample first, then reconcile labels/splits with official WLASL metadata before selecting the benchmark subset. Do not redistribute clips or derived data unless the applicable terms permit it.

The repository provides `download-wlasl-hf` (also available through `scripts\download_wlasl_hf.py`). It loads a bounded sample with FiftyOne, copies the media into the project's raw data directory, and writes a CSV manifest that preserves each sample's metadata as JSON.

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
download-wlasl-hf
```

The default is **100 samples** for inspection. Adjust the sample limit, or download the full Hub copy only if you have checked the terms and have sufficient disk space:

```powershell
download-wlasl-hf --max-samples 250
download-wlasl-hf --all
```

Outputs:

- Videos: `data\raw\huggingface_wlasl\videos\`
- Sample metadata CSV: `data\processed\manifests\huggingface_wlasl_manifest.csv`

The manifest stores the sample ID, local relative video path, and the sample's remaining FiftyOne fields as JSON. The Hub card lists 11,980 samples; no full-dataset size estimate is published here, so check available space before `--all`. Downloaded data is ignored by Git. The current Hub metadata has no official train/validation/test split field, so do not treat the sample as having benchmark splits.

The current local 100-sample manifest contains 19 distinct glosses. It is a small extraction smoke-test sample, not a balanced WLASL100 subset or a verified official split. Confirm what split metadata is actually present before using a Hub sample for training.

The Hub card identifies the license as `other`, says academic and computational use only (no commercial use), and references the C-UDA. Read the agreement and confirm that the intended use complies before running the downloader. For official annotation semantics, the WLASL repository documents fields such as `gloss`, `video_id`, `split`, `signer_id`, source URL, and frame bounds. Check the upstream README for current instructions.

### MediaPipe landmark extraction

Use Python 3.11 or 3.12 for the pinned MediaPipe version. The extractor samples up to 32 frames uniformly from each clip and stores pose, left-hand, and right-hand landmarks in that order; face landmarks are omitted. It reads preprocessing defaults from `configs\landmarks.json`, which can be overridden with command-line options.

Each `.npz` feature file contains:

- `keypoints`: float32 array with shape `(sequence_length, 75, 3)`, in x/y/z order.
- `landmark_mask`: boolean array `(sequence_length, 75)`; false means the landmark was not detected and its coordinates remain zero.
- `frame_mask`: boolean array `(sequence_length,)`; false marks padding for clips shorter than the target length.
- `sample_id`, `gloss`, and `file_name`.

Run a short validation first, then process all manifest rows:

```powershell
extract-wlasl-landmarks --limit 3
extract-wlasl-landmarks
```

Outputs go to `data\processed\landmarks\`, with a status report at `data\processed\manifests\landmark_manifest.csv`. The report lists per-video failures; the command returns a failure exit status if any clip could not be processed. Use `--overwrite` to recompute existing feature files. Keep raw videos and derived features out of Git.

Check the generated files with `python scripts\check_landmarks.py`. The checker validates the array shapes and masks and reports detected landmark counts; it exits with an error if outputs are missing, malformed, or any rows failed.

## 3. End-to-end implementation flow

### Phase A — Inventory and protocol

1. Record the source URL, annotation filename/version, retrieval date, agreement, and download route.
2. Parse the official annotation JSON and inspect fields, unique glosses, split values, signer IDs, and per-class counts.
3. Select the WLASL100 class list using the release's definition; save it to a versioned project artifact.
4. Build a manifest with one row per expected sample: video ID, gloss, class index, official split, signer ID, source URL, frame bounds, and local status.
5. Audit missing, duplicate, unreadable, and out-of-subset videos. Report counts by split and class before preprocessing.

**Exit check:** every included example maps to one known class and one official split; the class list and missing-video report are saved.

### Phase B — Download and verify videos

1. Use the official download/request procedure, or load the checked Hugging Face copy into a separate inspection workflow.
2. Keep the original annotation metadata unchanged and record which video IDs were obtained.
3. Verify that each local video opens, has nonzero duration, and contains the annotated sign interval.
4. Keep a download/status manifest so interrupted downloads can be resumed without silently changing the sample set.
5. Do not place raw videos in Git.

**Exit check:** the manifest distinguishes expected, downloaded, failed/missing, and successfully decoded samples.

### Phase C — Landmark extraction and preprocessing

1. Decode the clip at the documented sampling rate and use the annotation's frame bounds where applicable. Take care: the WLASL documentation describes frame bounds in relation to decoding at 25 FPS.
2. Sample/pad or temporally resample to a fixed initial length of 32 frames. Save a valid-frame mask so padding is not mistaken for observed motion.
3. Extract the left hand (21), right hand (21), and pose (33) landmarks per frame with a pinned MediaPipe configuration.
4. Flatten in a fixed, documented order to 225 x/y/z values per frame. Store missing landmarks consistently (for example, zero coordinates plus a visibility/presence mask) and preserve that mask as an additional input or metadata.
5. Optionally normalize coordinates relative to a body reference such as the hip midpoint and shoulder width. Handle absent or near-zero reference distances without producing infinities or NaNs.
6. Save processed arrays, labels, sample IDs, splits, and preprocessing metadata. Keep a single authoritative label-to-index mapping.

**Exit check:** processed examples have expected shapes, finite values, stable labels/splits, and a recorded mapping back to source video IDs.

### Phase D — Baseline and Transformer

1. Implement a dataset loader that reads processed sequences and masks and returns inputs, class IDs, and sample IDs.
2. Train a small LSTM baseline first. Confirm the training loop, loss, metrics, checkpointing, and reproducibility on a small smoke-test subset.
3. Implement a Transformer encoder with positional information and padding-mask support.
4. Train both models on the exact same training data and compare using the same validation procedure.
5. Apply augmentation only to training data (possible experiments: temporal speed variation, temporal crop, or frame dropout). Keep the first comparison free of augmentation or use identical augmentation for both models.
6. Select checkpoints using validation performance. Do not tune on the test split.

**Exit check:** both models train and save reproducible checkpoints; validation results are generated from the same split.

### Phase E — Evaluation and reporting

1. Evaluate the selected checkpoints once on the held-out official test split.
2. Report Top-1 and Top-5 accuracy, macro precision, macro recall, macro F1, and a confusion matrix. Top-5 is meaningful for this multi-class task but should not replace Top-1.
3. Include class-wise results or discuss class imbalance and classes with few usable examples.
4. Report dataset coverage: expected/downloaded/decoded/processed counts by split and class.
5. Compare LSTM and Transformer results without claiming target values as achieved values.
6. Save configurations, logs, metrics, plots, and checkpoint identifiers for each run.

**Exit check:** the report can be reproduced from saved configuration and manifest, and test results have not influenced model selection.

## 4. Suggested folder structure

The following layout reflects the current extraction scaffold. Model, training, and evaluation folders exist for the next phase, but are not implemented yet.

```text
.
├── README.md
├── IMPLEMENTATION_GUIDE.md
├── pyproject.toml                # Package metadata and runtime dependencies
├── requirements.txt              # Editable project install
├── requirements-dev.txt          # Project install plus pytest
├── .gitignore
├── configs/
│   └── landmarks.json
├── data/                         # Local data; do not commit
│   ├── annotations/              # Local copies of upstream annotation files
│   ├── raw/huggingface_wlasl/videos/
│   ├── interim/                  # Temporary decode and dataset-audit artifacts
│   └── processed/
│       ├── landmarks/            # Compressed NumPy arrays
│       └── manifests/
│           ├── huggingface_wlasl_manifest.csv
│           └── landmark_manifest.csv
├── src/
│   └── sign_language/
│       ├── __init__.py
│       ├── data/
│       │   ├── download.py
│       │   ├── manifest.py
│       │   └── landmarks.py
│       ├── models/               # __init__.py; next phase
│       ├── training/             # __init__.py; next phase
│       └── evaluation/           # __init__.py; next phase
├── scripts/
│   ├── download_wlasl_hf.py
│   ├── check_landmarks.py
│   └── extract_landmarks.py
├── tests/
│   ├── test_landmark_sampling.py
│   └── test_manifest.py
└── outputs/                      # Local experiment outputs; do not commit large files
    ├── checkpoints/
    ├── logs/
    ├── figures/
    └── results/
```

Keep module boundaries practical: an initial project can combine small files if that is clearer. Avoid creating empty placeholder modules just to match this tree.

## 5. Initial data and experiment artifacts

At minimum, keep these machine-readable files alongside each experiment (in a permitted location):

- `dataset_manifest.csv` or JSON: sample ID, gloss, class ID, official split, signer ID, local file/status.
- `class_to_idx.json`: stable mapping used by all models.
- Preprocessing configuration: frame sampling, clip bounds, landmark order, normalization, missing-point policy, target sequence length, MediaPipe version.
- Experiment configuration: seed, model, hidden dimensions/layers, optimizer, learning rate, batch size, epoch limit, checkpoint-selection metric.
- Results: split, sample count, Top-1/Top-5, macro precision/recall/F1, confusion matrix path, checkpoint path, and run/config identifier.

Do not silently drop failed clips. Record every exclusion and its reason, and ensure the final split counts reflect the samples actually used.

## 6. Git and reproducibility

The repository's `.gitignore` excludes local environments, downloaded media, derived features, and generated experiment outputs. Commit source code, configs, tests, and lightweight documentation—not videos, dataset-derived manifests, local tokens, or bulky checkpoints. Confirm that the data agreement permits sharing any derived artifact before publishing it.

```gitignore
.venv/
__pycache__/
*.py[cod]
data/raw/
data/interim/
data/processed/
outputs/checkpoints/
outputs/logs/
outputs/figures/
outputs/results/
```

Commit code, configs, tests, and lightweight documentation—not licensed video data, local tokens, or bulky checkpoints. If the data agreement permits sharing specific derived artifacts, document what they contain and confirm that sharing is allowed first. Record package versions and hardware for final experiments.

## 7. Semester milestones

1. **Protocol and acquisition:** confirm terms, inspect metadata, get a pilot subset, and establish the class/split manifest.
2. **Preprocessing:** decode clips, extract landmarks, validate arrays, and report coverage.
3. **Baseline:** implement and verify the LSTM training/evaluation path.
4. **Proposed model:** implement and compare the Transformer under the same protocol.
5. **Analysis and report:** run the final test evaluation, inspect errors, and document limitations and reproducibility details.

## References

- [Official WLASL repository](https://github.com/dxli94/WLASL) — annotations, downloader information, missing/preprocessed video request process, and C-UDA.
- [Voxel51 WLASL dataset card](https://huggingface.co/datasets/Voxel51/WLASL) — third-party Hub distribution; verify its terms and contents before use.
- [FiftyOne Hugging Face integration](https://docs.voxel51.com/integrations/huggingface.html) — loading datasets from the Hub.
- Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison.* WACV, 1459–1469.
