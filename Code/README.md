# SEM grain segmentation

U-Net++ with a ResNet18 encoder, squeeze-and-excitation attention on encoder
features, Tversky loss, and automatic morphology-based instance separation.

This revised implementation adds the methods described in the author's revision
response. It is not evidence that the original experiment used those methods.
Real results must be regenerated using the actual SEM dataset and film records.

## Project contents

```text
run.py                         train / evaluate / infer commands
configs/manuscript.json         manuscript settings
semseg/                        dataset, model, loss, metrics, workflow, post-processing
metadata/split_manifest.template.csv
scripts/validate_split.py       film-group and data checks
scripts/package_release.py      source-only ZIP packaging
tests/                         synthetic behavior and end-to-end tests
legacy/original_pipeline.py     original source, for comparison only
docs/                          manuscript statement and Chinese instructions
```

## Install

Use Python 3.10 or 3.11. This project was tested with Python 3.10.19 on Windows CPU.
Create an isolated environment, then install the dependencies:

```shell
python -m venv .venv
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.
On Linux/macOS, use `source .venv/bin/activate`. Then:

```shell
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU training, install the corresponding PyTorch/torchvision CUDA builds using
[PyTorch's official installation instructions](https://pytorch.org/get-started/locally/).
GPU training was not tested here. The complete CPU test environment is recorded
in `requirements-tested-cpu.txt`; it is a new verification environment, not the
original experiment environment. Training also saves its actual environment.

## Data and film-level split

The original SEM images and annotated masks are available from the corresponding
author upon reasonable request. **Authors must fill the contact details in
[data/README.md](data/README.md) before release.** Images and weights are not bundled.

Place the actual images and masks under `data/train`, `data/val`, and `data/test`,
each with `images/` and `masks/`. Fill
`metadata/split_manifest.csv` using the template header and real fabrication
records. Paths in the manifest are relative to `data/`.

The default configuration requires exactly 28/7/9 images and rejects film IDs
shared across subsets, duplicate image IDs/paths, missing or unlisted files, and
exact image-byte duplicates across subsets. The program checks supplied records;
it cannot authenticate film identities or identify every related crop.
Do not assign artificial film IDs to make a split pass.

```shell
python scripts/validate_split.py --manifest metadata/split_manifest.csv --data-root data
```

## Train

```shell
python run.py train --config configs/manuscript.json --data-root data --manifest metadata/split_manifest.csv --output-dir outputs/train_001
```

Default settings match the proposed method:

| Setting | Value |
| --- | --- |
| Original-image counts | 28 training / 7 validation / 9 test, checked from manifest |
| Input | 512 × 512 grayscale, repeated over 3 channels |
| Normalization | ImageNet mean 0.485/0.456/0.406, std 0.229/0.224/0.225 |
| Training augmentation | Independent horizontal/vertical flips, each probability 0.5; uniform 0/90/180/270° rotation |
| Image/mask pairing | Identical spatial transform; contiguous arrays |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Training | 50 epochs; batch size 2; seed 42; no learning-rate schedule |
| Loss | Tversky alpha 0.3 for FP, beta 0.7 for FN |
| Model selection | Highest per-image mean validation Dice, threshold 0.5 |
| Prediction | Single threshold 0.5 by default; automatic, no manual edits |

Augmentation runs only inside the training dataset and does not create more
independent samples. Validation and test images are never augmented. Training
does not score the test set. Metadata/file checks include all three subsets.
Requested ImageNet initialization must load successfully; errors do not silently
fall back to random weights. The program records the seed and environment but
does not promise bitwise identity across hardware/software environments.

Outputs include `best.pth`, `history.csv`, `best_validation_per_image.csv`,
`config.json`, `environment.json`, `data_audit.json`, and a manifest snapshot.
Use a new output directory for each run; existing output folders are not overwritten.

## Independent evaluation

```shell
python run.py evaluate --checkpoint outputs/train_001/best.pth --data-root data --manifest metadata/split_manifest.csv --split test --output-dir outputs/test_001
```

The model and all thresholds/post-processing settings come from the checkpoint.
The command does not accept a new configuration or choose a best test threshold.
Dataset/manifest fingerprints must match the training run. If tuning is needed,
use training/validation information before the final test and record the change.
Do not repeatedly select a run based on test scores.

Outputs: per-image and per-film Dice/IoU, summary JSON, probability maps,
semantic masks, integer instance labels, and overlay images. The primary score
is the mean across images; the per-film mean gives equal weight to each film.
Empty prediction and empty ground truth score 1. Dice/IoU measure the semantic
mask before instance splitting; they are not instance-level accuracy measures.

## Inference without annotations or retraining

```shell
python run.py infer --checkpoint outputs/train_001/best.pth --input path/to/sem_images --output-dir outputs/inference_001
```

Input may be one image or a directory. Outputs use numbered filenames and a CSV
mapping back to input names. Masks/probabilities are at the configured resized
resolution, not automatically restored to raw resolution.

Post-processing retains the original probability-gradient boundary cut,
connected components and local OpenCV watershed. All its parameters are in the
configuration and stored with the checkpoint. If selecting hysteresis in a
pre-test configuration, the inherited empty-high-seed fallback returns the low
threshold mask. Pixel-area thresholds refer to resized pixels. Physical grain
sizes require verified scale calibration and appropriate resizing/cropping treatment.

## Verification and release

```shell
python -m unittest discover -s tests -p test_pipeline.py -v
python tests/smoke.py
python scripts/package_release.py
```

Seven behavior tests, a one-epoch synthetic train/evaluate/infer run, dependency
checks, and a 512 × 512 model forward pass succeeded. The synthetic run used
64 × 64 images and no pretrained weights. The actual 44-image experiment,
ImageNet download, GPU operation, and scientific performance were not verified.
See [verification details](docs/verification.md).

[GitHub upload instructions (中文)](docs/github_setup_zh.md) ·
[Changes and remaining author inputs (中文)](docs/code_review_zh.md) ·
[Data and Code Availability](docs/data_and_code_availability.md)

No software license has been chosen in this local draft. The authors should
add the appropriate LICENSE and actual author/paper/contact details before
public release. No repository URL, paper DOI, trained-weight access, or research
performance is invented here.
