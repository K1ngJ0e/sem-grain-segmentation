# Verification of the revised implementation

Date: 2026-09-12. Environment: Windows, Python 3.10.19, CPU PyTorch 2.8.0+cpu,
torchvision 0.23.0+cpu, segmentation-models-pytorch 0.5.0, NumPy 1.26.4,
OpenCV 4.11.0.86. Full installed versions are in requirements-tested-cpu.txt.
This was a new isolated environment created to verify the revised code.

Completed checks:

- Seven behavior tests passed: paired pixel transformations; training-only
  augmentation; valid film manifest and cross-film leakage rejection; exact
  duplicate images across splits; correct per-image validation aggregation;
  finite Tversky gradients and empty-mask convention; explicit mask encodings.
- One-epoch synthetic training completed with 2 train / 3 validation / 1 test
  images at 64 × 64 and random initialization. The saved best checkpoint was
  reloaded for held-out evaluation and annotation-free inference. Automatic
  probability maps, masks, integer instance maps, and metrics were saved.
- A forward pass with a 1 × 3 × 512 × 512 input produced 1 × 1 × 512 × 512 logits.
- Dependency consistency check passed (`pip check`).

These are software checks only. No scientific performance estimate should be
taken from the synthetic data. The real 44-image dataset, author-supplied film
identities, original results, 50-epoch SEM training, ImageNet weight download,
CUDA behavior, and physical grain-size calibration have not been verified.

The smoke test intentionally changes input size, dataset counts, epoch count,
and pretrained setting. It does not change configs/manuscript.json. For the
manuscript experiment, use the actual data and the manuscript configuration,
then cite the resulting run records and corresponding source version.
