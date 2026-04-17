#!/usr/bin/env python3
"""Run the stage-3 preset patrol loop."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.stage3_preset_patrol import main


if __name__ == "__main__":
    main()
