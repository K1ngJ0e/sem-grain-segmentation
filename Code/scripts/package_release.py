"""Build a source-only GitHub ZIP; never include local data, weights, or environments."""

import argparse
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root.parent / "sem-grain-segmentation-github.zip")
    args = parser.parse_args()
    filenames = ["README.md", "run.py", "requirements.txt", "requirements-tested-cpu.txt", ".gitignore",
                 "data/README.md", "metadata/split_manifest.template.csv"]
    paths = [root / name for name in filenames]
    for folder, suffix in (("semseg", ".py"), ("configs", ".json"), ("scripts", ".py"),
                           ("tests", ".py"), ("docs", ".md"), ("legacy", ".py"), ("legacy", ".md")):
        paths.extend(path for path in (root / folder).rglob("*" )
                     if path.is_file() and path.suffix == suffix and "__pycache__" not in path.parts)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    print(f"Created {args.output} with {len(set(paths))} source/documentation files.")


if __name__ == "__main__":
    main()
