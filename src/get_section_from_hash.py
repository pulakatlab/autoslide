"""
If given a hash, return the section from the SVS corresponding to that hash.
"""

auto_slide_dir = '/media/bigdata/projects/auto_slide'
# auto_slide_dir = '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'

import os
import sys
import uuid
sys.path.append(os.path.join(auto_slide_dir, 'src'))
from pipeline import utils

import slideio
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np 
from pprint import pprint
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
import hashlib
from ast import literal_eval

def get_section_from_hash(hash_value, df):
    """
    Get the section from the dataframe corresponding to the hash value.
    
    Args:
        hash_value: str, the hash value to search for.
        df: pd.DataFrame, the dataframe containing the suggested regions.
        
    Returns:
        section: pd.Series or None, the section corresponding to the hash value.
    """
    section = df[df['section_hash'] == hash_value]
    if not section.empty:
        return section.iloc[0]
    else:
        return None

##############################
##############################

data_dir = os.path.join(auto_slide_dir, 'data')
tracking_dir = os.path.join(data_dir, '.tracking')

file_list = os.listdir(tracking_dir)
json_path_list = glob(os.path.join(tracking_dir, '*.json'))
json_list = [json.load(open(x, 'r')) for x in json_path_list]

suggested_regions_paths = []
basename_list = []
data_path_list = []
for x in json_list:
    try:
        suggested_regions_paths.append(x['suggested_regions_frame_path'])
        basename_list.append(x['file_basename'].split('.')[0])
        data_path_list.append(x['data_path'])
    except KeyError:
        print(f"KeyError in {x['file_basename']}, skipping...")
        continue

suggested_regions_list = []
for i in tqdm(range(len(suggested_regions_paths))):
    path = suggested_regions_paths[i]
    basename = basename_list[i]
    data_path = data_path_list[i]
    try:
        df = pd.read_csv(path) 
        df['basename'] = basename
        df['data_path'] = data_path
        suggested_regions_list.append(df)
    except FileNotFoundError:
        print(f"FileNotFoundError: {path}, skipping...")
        continue

final_df = pd.concat(suggested_regions_list, ignore_index=True)

##############################
test_hash = '0cb8cf88e2d3c22d'
section = get_section_from_hash(test_hash, final_df)
section_bounds = literal_eval(section['section_bounds'])

##############################

slide = slideio.open_slide(section['data_path'], 'SVS')
scene = slide.get_scene(0)

fig, ax = utils.visualize_sections(
        scene,
        [section_bounds],
        plot_n = -1,
        edgecolor = 'orange',
        return_image = True,
        crop_to_sections= True,
        down_sample = 10,
        )
plt.show()
