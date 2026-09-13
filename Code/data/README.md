# Dataset access and preparation

The original SEM images and annotated masks are available from the corresponding
author upon reasonable request. No real images, annotations, or trained
checkpoints are included in this project.

Corresponding author contact: **TO BE FILLED IN BY THE AUTHORS**.

## Layout

```text
data/
  train/images/   train/masks/
  val/images/     val/masks/
  test/images/    test/masks/
```

The default manuscript configuration requires 28 training, 7 validation, and
9 test images. Put each actual image and its mask in the selected subset.
The same independently fabricated film must not appear in multiple subsets.
Keep related crops and offline augmented copies with their source film.
The code adds online augmentation only after selecting a training image.

## Real split manifest

Fill `metadata/split_manifest.csv` using the supplied header:
`image_id,film_id,split,image_path,mask_path`.

Use unique image identifiers, true film identifiers from fabrication/imaging
records, and `train` / `val` / `test` split names. Paths are relative to this
directory, for example `train/images/<actual-filename>.png`.
Matching image/mask filename stems are required. Do not use duplicate names or
multiple annotations for one image. The header-only template contains no real
sample assignments and intentionally fails validation.

`scripts/validate_split.py` checks submitted records, counts, grouping, file
coverage, and exact byte duplicates. It does not prove the authenticity of film
identities or detect every transformed/cropped duplicate. Report both image
counts and actual film counts in the SI.

## Image and mask format

Supported extensions are PNG, JPG/JPEG, and TIF/TIFF. Lossless PNG/TIFF annotations
are preferable to JPEG artifacts. Images are loaded as 8-bit grayscale; assess
any 16-bit SEM conversion before use. The default bottom crop is 0, so scale
bars are not removed automatically.

For `mask_mode: red_on_black`, use color masks with red foreground and black
background. All-black masks are allowed. A nonblack mask containing no detected
red raises an error instead of silently applying Otsu thresholding.
For true grayscale binary annotations, select `mask_mode: binary` and use
0/1 or 0/255. Image and mask dimensions must match before resizing.

Images are resized with bilinear interpolation; masks use nearest neighbor.
Training transforms image and mask identically. Normalization uses fixed
ImageNet statistics, not statistics calculated from the complete dataset.

Raw files under `data/` and the filled internal manifest are excluded from
normal Git commits. Browser uploads do not enforce the local ignore rules.
Publish only the files actually intended for public release.
