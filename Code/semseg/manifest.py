"""Audit author-supplied split metadata; never infer or create film identities."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIELDS = {"image_id", "film_id", "split", "image_path", "mask_path"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(manifest, data_root, expected):
    data_root = data_root.resolve()
    errors = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not FIELDS.issubset(reader.fieldnames or []):
            return ["Manifest must have columns: " + ", ".join(sorted(FIELDS))], {}
        rows = list(reader)

    counts = Counter()
    films = defaultdict(set)
    ids = set()
    seen_paths = {"image_path": set(), "mask_path": set()}
    hashes = {}
    for line_number, row in enumerate(rows, start=2):
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in FIELDS):
            errors.append(f"Line {line_number}: missing field value")
            continue
        row = {key: row[key].strip() for key in FIELDS}
        split = row["split"]
        if split not in expected:
            errors.append(f"Line {line_number}: invalid split {split!r}")
            continue
        counts[split] += 1
        films[row["film_id"]].add(split)
        if row["image_id"] in ids:
            errors.append(f"Line {line_number}: duplicate image_id {row['image_id']!r}")
        ids.add(row["image_id"])

        for field, folder in (("image_path", "images"), ("mask_path", "masks")):
            relative = Path(row[field])
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"Line {line_number}: {field} must be a relative path under data_root")
                continue
            path = (data_root / relative).resolve()
            parent = (data_root / split / folder).resolve()
            if not parent.is_relative_to(data_root) or path.parent != parent:
                errors.append(f"Line {line_number}: {field} must be directly under {split}/{folder}")
                continue
            if path in seen_paths[field]:
                errors.append(f"Line {line_number}: duplicate {field}: {row[field]}")
            seen_paths[field].add(path)
            if not path.is_file():
                errors.append(f"Line {line_number}: missing file: {row[field]}")
                continue
            if path.suffix.lower() not in EXTENSIONS:
                errors.append(f"Line {line_number}: unsupported extension: {row[field]}")
            if field == "image_path":
                digest = file_hash(path)
                if digest in hashes and hashes[digest][0] != split:
                    errors.append(f"Exact image bytes shared across splits: {hashes[digest][1]} and {row[field]}")
                hashes.setdefault(digest, (split, row[field]))

        if Path(row["image_path"]).stem != Path(row["mask_path"]).stem:
            errors.append(f"Line {line_number}: image/mask stems do not match the source loader's convention")

    for split, count in expected.items():
        if counts[split] != count:
            errors.append(f"{split}: expected {count} images, found {counts[split]}")
        for field, folder in (("image_path", "images"), ("mask_path", "masks")):
            directory = data_root / split / folder
            if not directory.is_dir():
                errors.append(f"Missing folder: {directory}")
                continue
            actual = {path.resolve() for path in directory.iterdir()
                      if path.is_file() and path.suffix.lower() in EXTENSIONS}
            extra = actual - seen_paths[field]
            if extra:
                errors.append(f"{split}/{folder}: {len(extra)} image-format file(s) missing from manifest")

    for film, splits in films.items():
        if len(splits) > 1:
            errors.append(f"Film {film!r} appears in multiple splits: {', '.join(sorted(splits))}")
    summary = {
        "images": {split: counts[split] for split in expected},
        "films": {split: sum(split in splits for splits in films.values()) for split in expected},
    }
    return errors, summary


def validated_records(manifest, data_root, expected):
    errors, summary = audit(manifest, data_root, expected)
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors))
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        rows = [{key: value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    digest = hashlib.sha256()
    digest.update(Path(manifest).read_bytes())
    for row in sorted(rows, key=lambda row: row["image_id"]):
        for field in ("image_path", "mask_path"):
            digest.update(file_hash(Path(data_root) / row[field]).encode("ascii"))
    return rows, summary, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--expected-counts", nargs=3, type=int, default=(28, 7, 9),
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    if any(value < 0 for value in args.expected_counts):
        parser.error("Expected counts cannot be negative")
    try:
        errors, summary = audit(args.manifest, args.data_root,
                                dict(zip(("train", "val", "test"), args.expected_counts)))
    except (OSError, csv.Error) as exc:
        parser.exit(1, f"Could not audit data: {exc}\n")
    print(summary)
    if errors:
        for error in errors:
            print("ERROR:", error)
        raise SystemExit(1)
    print("PASS: submitted metadata and file checks passed.")
    print("This does not authenticate film IDs or rule out all related crops/augmented copies.")


if __name__ == "__main__":
    main()
