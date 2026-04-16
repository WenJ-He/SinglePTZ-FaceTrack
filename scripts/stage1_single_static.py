#!/usr/bin/env python3
"""Run the refactored stage-1 single-static loop."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.stage1_single_static import main


if __name__ == "__main__":
    main()
