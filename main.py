"""Main entry point for the video captioning system."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.cli import main

if __name__ == "__main__":
    main()
