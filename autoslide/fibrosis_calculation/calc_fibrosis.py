"""
Functionality to calculate fibrosis given a single image + optional mask
"""

import os
import sys
import slideio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
from ast import literal_eval
from autoslide import config
from autoslide.pipeline import utils
from autoslide.utils.get_section_from_hash import *


##############################
data_dir = config['data_dir']
tracking_dir = os.path.join(data_dir, 'tracking')

# Load tracking data
suggested_regions_paths, basename_list, data_path_list = load_tracking_data(
    tracking_dir)

# Load suggested regions
final_df = load_suggested_regions(
    suggested_regions_paths, basename_list, data_path_list)

# Test with a specific hash
test_hash = '0cb8cf88e2d3c22d'
section = get_section_from_hash(test_hash, final_df)

section_bounds = literal_eval(section['section_bounds'])
slide = slideio.open_slide(section['data_path'], 'SVS')
scene = slide.get_scene(0)

section = utils.get_section(scene, section_bounds, 1)

if section is not None:
    # Visualize the section
    # fig, ax = visualize_section(section, utils)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(section)
    fig.suptitle(f"Section from SVS\nSection: {section['basename']}, Hash: {section['section_hash']}" +
                 f"\nData Path: {section['data_path']},\nSection Bounds: {section['section_bounds']}")
    plt.tight_layout()
    plt.show()
else:
    print(f"No section found with hash {test_hash}")
