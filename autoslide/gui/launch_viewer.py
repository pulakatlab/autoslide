#!/usr/bin/env python
"""
Simple launcher script for the AutoSlide Section Viewer GUI
"""

import sys
import os

# Add the parent directory to the path so we can import autoslide
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autoslide.gui.section_viewer import main

if __name__ == "__main__":
    main()
