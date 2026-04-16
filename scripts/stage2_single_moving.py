#!/usr/bin/env python3
"""Run the stage-2 single-moving follow loop."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.stage2_single_moving import main


if __name__ == "__main__":
    main()
