"""One-epoch 64x64 synthetic end-to-end check. Not a paper experiment."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from test_pipeline import make_fixture
from semseg import engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="synthetic_smoke_", dir=args.output_root))
    _, rows, manifest, config_path = make_fixture(root)
    torch.set_num_threads(2)
    engine.train(SimpleNamespace(config=config_path, manifest=manifest, data_root=root / "data",
                                 output_dir=root / "train", device="cpu"))
    checkpoint = root / "train" / "best.pth"
    engine.evaluate(SimpleNamespace(checkpoint=checkpoint, manifest=manifest, data_root=root / "data",
                                    output_dir=root / "test", device="cpu", split="test"))
    engine.infer(SimpleNamespace(checkpoint=checkpoint,
                                 input=root / "data" / rows[-1]["image_path"],
                                 output_dir=root / "inference", device="cpu"))
    metrics = json.loads((root / "test" / "metrics.json").read_text())
    assert metrics["n_images"] == 1 and metrics["manual_correction"] is False
    assert (root / "inference" / "image_001_instances.npy").is_file()
    print(f"SYNTHETIC SMOKE PASS: {root}; not manuscript data or performance.")


if __name__ == "__main__":
    main()
