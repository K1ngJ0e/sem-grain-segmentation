"""Revised SEM workflow: train, evaluate, infer. Python 3.10+."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True,
                             help="New output directory; existing directories are not overwritten")
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name == "train":
            command.add_argument("--config", type=Path, default=Path("configs/manuscript.json"))
        else:
            command.add_argument("--checkpoint", type=Path, required=True)
        if name in ("train", "evaluate"):
            command.add_argument("--data-root", type=Path, required=True)
            command.add_argument("--manifest", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=("val", "test"), default="test")
        if name == "infer":
            command.add_argument("--input", type=Path, required=True, help="Image file or image directory")
    args = parser.parse_args()
    from semseg import engine
    {"train": engine.train, "evaluate": engine.evaluate, "infer": engine.infer}[args.command](args)


if __name__ == "__main__":
    main()
