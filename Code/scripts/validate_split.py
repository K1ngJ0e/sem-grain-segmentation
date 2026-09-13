"""Standalone metadata audit; Python 3.10+; no ML dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semseg.manifest import main

if __name__ == "__main__":
    main()
