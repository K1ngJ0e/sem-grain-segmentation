"""Train, evaluate fixed checkpoints, and infer without manual correction."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SEMDataset, load_image, resize_image, to_tensor
from .manifest import EXTENSIONS, file_hash, validated_records
from .metrics import binary_metrics, tversky_loss
from .model import UnetPlusPlusEncoderSE
from .postprocess import (binarize_hysteresis, binarize_single,
                          overlay_instance_boundaries_morph, overlay_instances_alpha,
                          split_instances_hybrid_morph)


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No records to write")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    height, width = config["data"]["target_size"]
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise ValueError("Input dimensions must be positive multiples of 32")
    if config["augmentation"]["enabled"] and config["augmentation"]["rotate_90"] and height != width:
        raise ValueError("90-degree augmentation requires square targets in this pipeline")
    for field in ("horizontal_flip_probability", "vertical_flip_probability"):
        if not 0 <= config["augmentation"][field] <= 1:
            raise ValueError(f"Invalid augmentation probability: {field}")
    training = config["training"]
    if training["epochs"] < 1 or training["batch_size"] < 1 or training["learning_rate"] <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    if config["prediction"]["method"] not in ("single", "hysteresis"):
        raise ValueError("prediction.method must be single or hysteresis")
    if not 0 <= config["prediction"]["hysteresis_low"] <= config["prediction"]["hysteresis_high"] <= 1:
        raise ValueError("Invalid hysteresis thresholds")
    for value in (training["selection_threshold"], config["prediction"]["threshold"]):
        if not 0 <= value <= 1:
            raise ValueError("Binary thresholds must be between 0 and 1")
    return config


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    return torch.device(name)


def loader_for(dataset, config, shuffle, device):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle,
                      num_workers=config["training"]["num_workers"],
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=generator)


def environment_info(device):
    packages = {}
    for name in ("torch", "torchvision", "segmentation-models-pytorch", "numpy", "opencv-python", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "device": str(device), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(list((root / "semseg").glob("*.py")) + [root / "run.py"]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@torch.no_grad()
def validation_records(model, loader, device, threshold):
    model.eval()
    records = []
    for batch in loader:
        probabilities = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
        targets = batch["mask"].numpy()[:, 0]
        for index, (probability, target) in enumerate(zip(probabilities, targets)):
            dice, iou = binary_metrics(probability >= threshold, target)
            records.append({"image_id": batch["image_id"][index], "film_id": batch["film_id"][index],
                            "dice": dice, "iou": iou})
    return records


def summarize(records):
    if not records:
        raise ValueError("Cannot score an empty split")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["film_id"]].append(row)
    films = [{"film_id": film, "n_images": len(rows),
              "dice": float(np.mean([row["dice"] for row in rows])),
              "iou": float(np.mean([row["iou"] for row in rows]))}
             for film, rows in sorted(grouped.items())]
    result = {"n_images": len(records), "n_films": len(films),
              "image_mean_dice": float(np.mean([row["dice"] for row in records])),
              "image_mean_iou": float(np.mean([row["iou"] for row in records])),
              "film_mean_dice": float(np.mean([row["dice"] for row in films])),
              "film_mean_iou": float(np.mean([row["iou"] for row in films])),
              "manual_correction": False,
              "empty_mask_convention": "empty prediction and empty target score 1"}
    return result, films


def train(args):
    config = read_config(args.config)
    rows, audit_summary, fingerprint = validated_records(
        args.manifest, args.data_root, config["data"]["expected_counts"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = get_device(args.device)
    seed_everything(config["seed"])
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment_info(device))
    write_json(output / "data_audit.json", {"summary": audit_summary, "data_sha256": fingerprint})
    write_csv(output / "split_manifest.csv", rows)
    training = config["training"]
    training_data = SEMDataset(rows, args.data_root, "train", config)
    validation_data = SEMDataset(rows, args.data_root, "val", config)
    train_loader = loader_for(training_data, config, True, device)
    val_loader = loader_for(validation_data, config, False, device)
    model = UnetPlusPlusEncoderSE(pretrained=config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    best_dice, history = -1.0, []
    code_sha256 = source_hash()
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = tversky_loss(model(inputs), targets, training["alpha"], training["beta"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.shape[0]
        records = validation_records(model, val_loader, device, training["selection_threshold"])
        summary, _ = summarize(records)
        dice = summary["image_mean_dice"]
        if dice > best_dice:
            best_dice = dice
            torch.save({"format_version": 1, "state_dict": model.state_dict(), "epoch": epoch,
                        "best_val_dice": best_dice, "config": config, "data_sha256": fingerprint,
                        "source_sha256": code_sha256}, output / "best.pth")
            write_csv(output / "best_validation_per_image.csv", records)
        history.append({"epoch": epoch, "train_loss": total_loss / len(training_data),
                        "validation_dice": dice, "best_validation_dice": best_dice})
        write_csv(output / "history.csv", history)
        print(f"epoch {epoch:03d} | loss {history[-1]['train_loss']:.5f} | val Dice {dice:.5f}", flush=True)
    print(f"Best checkpoint: {output / 'best.pth'}; test set has not been scored.")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Use a checkpoint produced by this revised pipeline; legacy weights lack run metadata")
    model = UnetPlusPlusEncoderSE(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def predict_mask(probability, settings):
    if settings["method"] == "single":
        return binarize_single(probability, settings["threshold"])
    return binarize_hysteresis(probability, settings["hysteresis_low"], settings["hysteresis_high"])


def save_prediction(output, name, raw, probability, mask, config):
    labels, count, _ = split_instances_hybrid_morph(probability, mask, **config["postprocess"])
    np.save(output / f"{name}_probability.npy", probability.astype(np.float32))
    np.save(output / f"{name}_instances.npy", labels.astype(np.int32))
    artifacts = {
        "mask": mask * 255,
        "instances_overlay": cv2.cvtColor(overlay_instances_alpha(raw, labels), cv2.COLOR_RGB2BGR),
        "boundaries": cv2.cvtColor(overlay_instance_boundaries_morph(raw, labels), cv2.COLOR_RGB2BGR),
    }
    for suffix, value in artifacts.items():
        if not cv2.imwrite(str(output / f"{name}_{suffix}.png"), value):
            raise OSError("Could not save prediction image")
    return count


@torch.no_grad()
def evaluate(args):
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    rows, _, fingerprint = validated_records(args.manifest, args.data_root, config["data"]["expected_counts"])
    if fingerprint != checkpoint["data_sha256"]:
        raise ValueError("Manifest/data differ from the training run; refusing an inconsistent held-out evaluation")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = SEMDataset(rows, args.data_root, args.split, config)
    loader = loader_for(dataset, config, False, device)
    records = []
    for batch in loader:
        probabilities = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
        targets = batch["mask"].numpy()[:, 0]
        for index, (probability, target) in enumerate(zip(probabilities, targets)):
            mask = predict_mask(probability, config["prediction"])
            dice, iou = binary_metrics(mask, target)
            count = save_prediction(output, f"image_{len(records) + 1:03d}",
                                    batch["raw"][index].numpy(), probability, mask, config)
            records.append({"image_id": batch["image_id"][index], "film_id": batch["film_id"][index],
                            "artifact_prefix": f"image_{len(records) + 1:03d}",
                            "dice": dice, "iou": iou, "instance_count": count})
    summary, films = summarize(records)
    summary.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"],
                    "checkpoint_sha256": file_hash(args.checkpoint), "data_sha256": fingerprint,
                    "training_source_sha256": checkpoint["source_sha256"],
                    "evaluation_source_sha256": source_hash(), "prediction": config["prediction"],
                    "postprocess": config["postprocess"]})
    write_csv(output / "per_image.csv", records)
    write_csv(output / "per_film.csv", films)
    write_json(output / "metrics.json", summary)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment_info(device))
    print(json.dumps(summary, indent=2))


@torch.no_grad()
def infer(args):
    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    source = Path(args.input)
    paths = [source] if source.is_file() else sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError("No input images found")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, path in enumerate(paths, start=1):
        raw = resize_image(load_image(path), config["data"])
        probability = model(to_tensor(raw).unsqueeze(0).to(device)).sigmoid()[0, 0].cpu().numpy()
        mask = predict_mask(probability, config["prediction"])
        prefix = f"image_{index:03d}"
        count = save_prediction(output, prefix, raw, probability, mask, config)
        records.append({"input_file": path.name, "artifact_prefix": prefix, "instance_count": count})
    write_csv(output / "predictions.csv", records)
    write_json(output / "config.json", config)
    write_json(output / "run.json", {"checkpoint_sha256": file_hash(args.checkpoint),
                                     "manual_correction": False, "source_sha256": source_hash()})
    write_json(output / "environment.json", environment_info(device))
    print(f"Saved {len(records)} automatic predictions to {output}")
