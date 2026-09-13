"""Paired SEM data loading; stochastic spatial augmentation is training-only."""

import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read SEM image: {path}")
    return image


def load_mask(path, mode):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read annotation: {path}")
    if mode == "red_on_black":
        if mask.ndim != 3 or mask.shape[2] not in (3, 4):
            raise ValueError(f"Expected a red-on-black color mask: {path}; use binary for grayscale masks")
        hsv = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red |= cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        result = (red > 0).astype(np.uint8)
        if not result.any() and mask[:, :, :3].max() > 0:
            raise ValueError(f"Nonblack mask contains no red foreground: {path}; verify mask_mode")
        return result
    if mode == "binary":
        if mask.ndim == 3:
            if not (np.array_equal(mask[:, :, 0], mask[:, :, 1]) and
                    np.array_equal(mask[:, :, 0], mask[:, :, 2])):
                raise ValueError(f"Binary mask channels are not grayscale: {path}")
            mask = mask[:, :, 0]
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"Binary mask must use 0/1 or 0/255 values: {path}")
        return (mask > 0).astype(np.uint8)
    raise ValueError(f"Unsupported mask mode: {mode}")


def spatial_augment(image, mask, settings, rng=random):
    if rng.random() < settings["horizontal_flip_probability"]:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    if rng.random() < settings["vertical_flip_probability"]:
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    if settings["rotate_90"]:
        turns = rng.randrange(4)
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def resize_image(image, settings):
    crop = settings["crop_bottom"]
    if crop < 0 or crop >= image.shape[0]:
        raise ValueError("crop_bottom must leave at least one row")
    if crop:
        image = image[:-crop, :]
    height, width = settings["target_size"]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def to_tensor(image):
    channels = np.repeat((image.astype(np.float32) / 255.0)[None], 3, axis=0)
    return torch.from_numpy(np.ascontiguousarray((channels - MEAN) / STD))


class SEMDataset(Dataset):
    def __init__(self, rows, root, split, config):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        self.rows = [row for row in rows if row["split"] == split]
        self.root = Path(root)
        self.split = split
        self.settings = config["data"]
        self.augmentation = config["augmentation"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_image(self.root / row["image_path"])
        mask = load_mask(self.root / row["mask_path"], self.settings["mask_mode"])
        if image.shape != mask.shape:
            raise ValueError(f"Image and annotation dimensions differ: {row['image_id']}")
        image = resize_image(image, self.settings)
        crop = self.settings["crop_bottom"]
        if crop:
            mask = mask[:-crop, :]
        height, width = self.settings["target_size"]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.split == "train" and self.augmentation["enabled"]:
            image, mask = spatial_augment(image, mask, self.augmentation)
        return {
            "image": to_tensor(image),
            "mask": torch.from_numpy(mask.astype(np.float32)[None]),
            "raw": torch.from_numpy(image.copy()),
            "image_id": row["image_id"], "film_id": row["film_id"],
        }
