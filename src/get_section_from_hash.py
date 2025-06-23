"""
If given a hash, return the section from the SVS corresponding to that hash.
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

def get_auto_slide_dir():
    """Get the auto_slide directory path based on the environment."""
    # Check for different possible paths
    if os.path.exists('/media/bigdata/projects/auto_slide'):
        return '/media/bigdata/projects/auto_slide'
    elif os.path.exists('/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'):
        return '/home/abuzarmahmood/projects/pulakat_lab/auto_slide/'
    else:
        raise FileNotFoundError("Could not find auto_slide directory")

def setup_environment():
    """Setup the environment by adding the src directory to the path."""
    auto_slide_dir = get_auto_slide_dir()
    sys.path.append(os.path.join(auto_slide_dir, 'src'))
    from pipeline import utils
    return auto_slide_dir, utils

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

def load_tracking_data(tracking_dir):
    """
    Load tracking data from JSON files.
    
    Args:
        tracking_dir: str, path to the tracking directory.
        
    Returns:
        suggested_regions_paths: list, paths to suggested regions CSV files.
        basename_list: list, basenames of the files.
        data_path_list: list, paths to the data files.
    """
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
            
    return suggested_regions_paths, basename_list, data_path_list

def load_suggested_regions(suggested_regions_paths, basename_list, data_path_list):
    """
    Load suggested regions from CSV files.
    
    Args:
        suggested_regions_paths: list, paths to suggested regions CSV files.
        basename_list: list, basenames of the files.
        data_path_list: list, paths to the data files.
        
    Returns:
        final_df: pd.DataFrame, concatenated dataframe of all suggested regions.
    """
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
            
    return pd.concat(suggested_regions_list, ignore_index=True)

def visualize_section(section, utils):
    """
    Visualize a section from a slide.
    
    Args:
        section: pd.Series, the section to visualize.
        utils: module, the utils module.
        
    Returns:
        fig, ax: matplotlib figure and axes.
    """
    section_bounds = literal_eval(section['section_bounds'])
    
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
    
    return fig, ax

def main():
    """Main function to run the script."""
    # Setup environment
    auto_slide_dir, utils = setup_environment()
    
    # Define paths
    data_dir = os.path.join(auto_slide_dir, 'data')
    tracking_dir = os.path.join(data_dir, '.tracking')
    
    # Load tracking data
    suggested_regions_paths, basename_list, data_path_list = load_tracking_data(tracking_dir)
    
    # Load suggested regions
    final_df = load_suggested_regions(suggested_regions_paths, basename_list, data_path_list)
    
    # Test with a specific hash
    test_hash = '0cb8cf88e2d3c22d'
    section = get_section_from_hash(test_hash, final_df)
    
    if section is not None:
        # Visualize the section
        fig, ax = visualize_section(section, utils)
        plt.show()
    else:
        print(f"No section found with hash {test_hash}")

if __name__ == "__main__":
    main()
