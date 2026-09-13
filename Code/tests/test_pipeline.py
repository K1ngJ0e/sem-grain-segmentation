"""Behavior checks on synthetic data only; no manuscript performance claims."""

import copy
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from semseg.data import SEMDataset, load_mask, spatial_augment
from semseg.engine import summarize, validation_records
from semseg.manifest import audit, validated_records
from semseg.metrics import binary_metrics, tversky_loss


def make_fixture(root):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/manuscript.json").read_text())
    config["data"]["target_size"] = [64, 64]
    config["data"]["expected_counts"] = {"train": 2, "val": 3, "test": 1}
    config["training"]["epochs"] = 1
    config["model"]["pretrained"] = False
    rows = []
    rng = np.random.default_rng(24)
    for split, count in config["data"]["expected_counts"].items():
        for folder in ("images", "masks"):
            (root / "data" / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f"synthetic_{split}_{index}.png"
            raw = rng.integers(0, 80, size=(64, 64), dtype=np.uint8)
            raw[12:40, 8 + index:35 + index] = 200
            mask = np.zeros((64, 64, 3), dtype=np.uint8)
            mask[12:40, 8 + index:35 + index, 2] = 255
            image_path, mask_path = f"{split}/images/{name}", f"{split}/masks/{name}"
            assert cv2.imwrite(str(root / "data" / image_path), raw)
            assert cv2.imwrite(str(root / "data" / mask_path), mask)
            rows.append({"image_id": name, "film_id": f"synthetic_film_{split}_{index}",
                         "split": split, "image_path": image_path, "mask_path": mask_path})
    manifest = root / "manifest.csv"
    save_manifest(manifest, rows)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config, rows, manifest, config_path


def save_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.rows, self.manifest, _ = make_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_augmentation_keeps_every_pixel_paired(self):
        pattern = np.arange(81).reshape(9, 9)
        for seed in range(40):
            image, mask = spatial_augment(pattern, pattern.copy(), self.config["augmentation"], random.Random(seed))
            np.testing.assert_array_equal(image, mask)
            self.assertTrue(image.flags.c_contiguous)
        image, _ = spatial_augment(pattern, pattern.copy(),
                                   {"horizontal_flip_probability": 1, "vertical_flip_probability": 0,
                                    "rotate_90": False}, random.Random(0))
        np.testing.assert_array_equal(image, pattern[:, ::-1])

    def test_only_training_calls_augmentation(self):
        for split in ("val", "test"):
            dataset = SEMDataset(self.rows, self.root / "data", split, self.config)
            with patch("semseg.data.spatial_augment", side_effect=AssertionError("augmentation used")):
                first, second = dataset[0], dataset[0]
                torch.testing.assert_close(first["image"], second["image"])
        dataset = SEMDataset(self.rows, self.root / "data", "train", self.config)
        with patch("semseg.data.spatial_augment", wraps=spatial_augment) as mocked:
            dataset[0]
            self.assertEqual(mocked.call_count, 1)

    def test_manifest_passes_and_rejects_cross_film_leakage(self):
        _, summary, _ = validated_records(self.manifest, self.root / "data", self.config["data"]["expected_counts"])
        self.assertEqual(summary["images"], {"train": 2, "val": 3, "test": 1})
        rows = copy.deepcopy(self.rows)
        rows[-1]["film_id"] = rows[0]["film_id"]
        save_manifest(self.manifest, rows)
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validated_records(self.manifest, self.root / "data", self.config["data"]["expected_counts"])

    def test_duplicate_images_rejected_across_splits(self):
        first = self.root / "data" / self.rows[0]["image_path"]
        last = self.root / "data" / self.rows[-1]["image_path"]
        last.write_bytes(first.read_bytes())
        errors, _ = audit(self.manifest, self.root / "data", self.config["data"]["expected_counts"])
        self.assertTrue(any("Exact image bytes" in error for error in errors))

    def test_validation_mean_is_per_image_not_per_batch(self):
        class IdentityModel(torch.nn.Module):
            def forward(self, inputs):
                return inputs
        samples = [{"image": torch.full((1, 2, 2), value), "mask": torch.ones(1, 2, 2),
                    "image_id": str(index), "film_id": str(index)}
                   for index, value in enumerate((20.0, 20.0, -20.0))]
        records = validation_records(IdentityModel(), DataLoader(samples, batch_size=2), torch.device("cpu"), 0.5)
        summary, _ = summarize(records)
        self.assertAlmostEqual(summary["image_mean_dice"], 2 / 3, places=6)

    def test_loss_gradients_and_empty_masks(self):
        logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 1:3, 1:3] = 1
        loss = tversky_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        blank = np.zeros((4, 4))
        self.assertEqual(binary_metrics(blank, blank), (1.0, 1.0))

    def test_wrong_mask_encoding_is_not_silently_thresholded(self):
        path = self.root / "white.png"
        cv2.imwrite(str(path), np.full((8, 8, 3), 255, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "no red"):
            load_mask(path, "red_on_black")
        self.assertEqual(load_mask(path, "binary").sum(), 64)


if __name__ == "__main__":
    unittest.main()
