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
from autoslide import config
from autoslide.pipeline import utils


def get_section_details_from_hash(hash_value, df):
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


def get_section_from_hash(hash_value, df, down_sample=1):
    """
    Get the section from the dataframe corresponding to the hash value.

    Args:
        hash_value: str, the hash value to search for.
        df: pd.DataFrame, the dataframe containing the suggested regions.

    Returns:
        section: pd.Series or None, the section corresponding to the hash value.
    """
    section_details = get_section_details_from_hash(hash_value, df)

    section_bounds = literal_eval(section_details['section_bounds'])
    slide = slideio.open_slide(section_details['data_path'], 'SVS')
    scene = slide.get_scene(0)

    section = utils.get_section(scene, section_bounds,
                                down_sample=down_sample)

    return section, section_details


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
        plot_n=-1,
        edgecolor='orange',
        return_image=True,
        crop_to_sections=True,
        down_sample=10,
    )

    return fig, ax


def main():
    """Main function to run the script."""
    # Setup environment
    # auto_slide_dir, utils = setup_environment()

    # Define paths
    # data_dir = os.path.join(auto_slide_dir, 'data')
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
    section = get_section_details_from_hash(test_hash, final_df)

    if section is not None:
        # Visualize the section
        fig, ax = visualize_section(section, utils)
        fig.suptitle(f"Section from SVS\nSection: {section['basename']}, Hash: {section['section_hash']}" +
                     f"\nData Path: {section['data_path']},\nSection Bounds: {section['section_bounds']}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No section found with hash {test_hash}")

    ##############################
    # Make sure all current images are accounted for
    image_dirs = glob(os.path.join(
        data_dir, 'suggested_regions', '*', "images"))
    image_path_list = []
    for x in image_dirs:
        image_path_list.extend(glob(os.path.join(x, '*.png')))
    hash_list = [x.split('_')[-1].split('.')[0] for x in image_path_list]
    # Make sure all images have unique hashes
    unique_hashes = set(hash_list)
    assert len(unique_hashes) == len(
        hash_list), "There are duplicate hashes in the images."

    all_accounted = all(
        hash in final_df['section_hash'].values for hash in unique_hashes)
    if all_accounted:
        print("All images are accounted for in the final dataframe.")
    else:
        print("Some images are not accounted for in the final dataframe.")


if __name__ == "__main__":
    main()
