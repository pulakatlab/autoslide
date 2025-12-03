"""
Take annotated csv and images
Merge marked tissues and label appropriately with tissue type
"""

import pylab as plt
import numpy as np
import os
import json
import argparse
import time
from pprint import pprint
from glob import glob
import pandas as pd
from tqdm import tqdm
from skimage.color import label2rgb
from ast import literal_eval
from autoslide.src import config


def parse_args():
    parser = argparse.ArgumentParser(description='Process final annotations')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = args.verbose

    if verbose:
        print("Starting final annotation processing...")

    # Get directories from config
    data_dir = config['data_dir']
    init_annot_dir = config['initial_annotation_dir']
    file_list = os.listdir(init_annot_dir)

    if verbose:
        print(f"Data directory: {data_dir}")
        print(f"Initial annotation directory: {init_annot_dir}")
        print(f"Found {len(file_list)} files in initial annotation directory")

    fin_annotation_dir = config['final_annotation_dir']
    if not os.path.exists(fin_annotation_dir):
        os.makedirs(fin_annotation_dir)
        if verbose:
            print(f"Created final annotation directory: {fin_annotation_dir}")
    elif verbose:
        print(f"Final annotation directory exists: {fin_annotation_dir}")

    tracking_dir = config['tracking_dir']
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)
        if verbose:
            print(f"Created tracking directory: {tracking_dir}")
    elif verbose:
        print(f"Tracking directory exists: {tracking_dir}")

    # basenames = [os.path.basename(x).split('.')[0] for x in file_list]
    # unique_basenames = np.unique(basenames)

    json_path_list = glob(os.path.join(tracking_dir, '*.json'))
    json_list = [json.load(open(x, 'r')) for x in json_path_list]

    if verbose:
        print(f"Found {len(json_path_list)} JSON tracking files")
        for json_path in json_path_list:
            print(f"  - {json_path}")

    ############################################################

    # for this_basename in tqdm(unique_basenames):
    for this_json, json_path in tqdm(zip(json_list, json_path_list), total=len(json_list)):
        # this_basename = unique_basenames[0]
        this_basename = this_json['file_basename'].split('.')[0]
        
        # Start timing for this file
        start_time = time.time()

        if verbose:
            print(f"\nProcessing: {this_basename}")
            print(f"JSON path: {json_path}")

        metadata_path = this_json['wanted_regions_frame_path']
        # metadata = pd.read_csv(os.path.join(init_annot_dir, this_basename + '.csv'))
        metadata = pd.read_csv(metadata_path)

        if verbose:
            print(f"Metadata path: {metadata_path}")
            print(f"Metadata shape: {metadata.shape}")
            print(f"Tissue types found: {metadata['tissue_type'].unique()}")

        # Make sure tissue_num >0 as 0 is background
        assert np.all(metadata['tissue_num'] > 0), 'tissue_num should be >0'

        mask_path = this_json['initial_mask_path']
        # mask = np.load(os.path.join(init_annot_dir, this_basename + '.npy'))
        mask = np.load(mask_path)

        if verbose:
            print(f"Mask path: {mask_path}")
            print(f"Mask shape: {mask.shape}")
            print(f"Unique mask values: {np.unique(mask)}")

        label_map = {}
        for i, row in metadata.iterrows():
            label_map[row['label']] = row['tissue_num']

        if verbose:
            print(f"Label mapping: {label_map}")

        # Also get string to label each tissue
        metadata['tissue_str'] = metadata['tissue_num'].astype(
            str) + '_' + metadata['tissue_type']

        if verbose:
            print(f"Tissue strings: {metadata['tissue_str'].tolist()}")

        # Map values in mask according to label_map
        for key, value in label_map.items():
            mask[mask == key] = value

        if verbose:
            print(f"Mask values after mapping: {np.unique(mask)}")

        # Plot mask and also write to file
        image_label_overlay = label2rgb(mask,
                                        image=mask > 0,
                                        bg_label=0)

        fig, ax = plt.subplots(figsize=(5, 10))
        ax.imshow(image_label_overlay, cmap='tab10')
        ax.set_title(this_basename)
        # Label with tissue type
        for i, row in metadata.iterrows():
            # Centroid element could be of form: (np.float64(388.38771290085384), np.float64(431.56985248878283))
            # Extract using regex
            if isinstance(row['centroid'], str) and 'np.float64' in row['centroid']:
                row['centroid'] = row['centroid'].replace('np.float64', '')
                row['centroid'] = row['centroid'].replace('(', '')
                row['centroid'] = row['centroid'].replace(')', '')
            centroid = literal_eval(row['centroid'])
            ax.text(centroid[1], centroid[0],
                    row['tissue_str'], color='red',
                    fontsize=25, weight='bold')

        plot_path = os.path.join(fin_annotation_dir, this_basename + '.png')
        fig.savefig(plot_path)
        plt.close(fig)

        if verbose:
            print(f"Saved visualization: {plot_path}")

        fin_mask_path = os.path.join(
            fin_annotation_dir, this_basename + '.npy')
        np.save(fin_mask_path, mask)
        this_json['fin_mask_path'] = fin_mask_path

        if verbose:
            print(f"Saved final mask: {fin_mask_path}")

        # Record processing time
        end_time = time.time()
        processing_time = end_time - start_time
        this_json['final_annotation_processing_time'] = processing_time

        # Save the updated json with final mask path and processing time
        with open(json_path, 'w') as f:
            json.dump(this_json, f, indent=4)

        if verbose:
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Updated JSON file: {json_path}")

    if verbose:
        print(f"\nFinal annotation processing complete!")
        print(f"Processed {len(json_list)} files")


if __name__ == "__main__":
    main()
