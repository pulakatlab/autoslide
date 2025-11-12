"""
Iterate through image given window size and stride, and return a list of
regions to be used for classification.
"""

import os
import sys
import uuid
import slideio
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pprint import pprint
import pandas as pd
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes
from glob import glob
from autoslide import config
import json
from tqdm import tqdm
import hashlib
import argparse
import time

# Import utilities directly
from autoslide.pipeline import utils


def remove_mask_edge(
        mask,
        mask_resolution,
        closing_len=75e-6,  # meters
        edge_len=100e-6,  # meters
):
    """
    Remove the edge of the mask to avoid getting sections on the edge

    Args:
        mask: np.array
        mask_resolution: float
        closing_len: float
        edge_len: float

    Returns:
        mask: np.array
    """
    dilate_kern_len = int(np.round((closing_len / mask_resolution)))
    edge_kern_len = int(np.ceil((edge_len / mask_resolution)))
    dilate_kern = morph.disk(dilate_kern_len)
    erode_kern = morph.disk(edge_kern_len*2)

    dilated_mask = morph.binary_dilation(mask, dilate_kern)
    filled_mask = morph.erosion(dilated_mask, dilate_kern)
    eroded_mask = morph.erosion(filled_mask, erode_kern)

    return eroded_mask


def str_to_hash(s):
    """
    Convert a string to a hash value.

    Args:
        s (str): The input string.

    Returns:
        str: A hexadecimal hash value of the input string.
    """
    return hashlib.sha256(s.encode()).hexdigest()[:16]  # Shorten to 16 characters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Suggest regions for classification')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = args.verbose
    
    if verbose:
        print("Starting region suggestion pipeline...")
        print(f"Verbose mode enabled")
    
    # Get directories from config
    data_dir = config['data_dir']
    mask_dir = os.path.join(data_dir, 'final_annotation')
    metadata_dir = os.path.join(data_dir, 'initial_annotation')
    output_base_dir = os.path.join(data_dir, 'suggested_regions')
    
    if verbose:
        print(f"Data directory: {data_dir}")
        print(f"Mask directory: {mask_dir}")
        print(f"Metadata directory: {metadata_dir}")
        print(f"Output base directory: {output_base_dir}")
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir, exist_ok=True)
        if verbose:
            print(f"Created output directory: {output_base_dir}")
    
    tracking_dir = os.path.join(data_dir, 'tracking')
    if verbose:
        print(f"Tracking directory: {tracking_dir}")

    file_list = os.listdir(tracking_dir)
    json_path_list = glob(os.path.join(tracking_dir, '*.json'))
    json_list = [json.load(open(x, 'r')) for x in json_path_list]
    
    if verbose:
        print(f"Found {len(json_path_list)} JSON files to process")
        for i, json_path in enumerate(json_path_list):
            print(f"  {i+1}: {os.path.basename(json_path)}")

    ############################################################

    # for data_path in data_path_list:
    for i, (this_json, json_path) in enumerate(tqdm(zip(json_list, json_path_list), total=len(json_list))):
        data_path = this_json['data_path']
        
        # Start timing for this file
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(json_list)}: {os.path.basename(json_path)}")
            print(f"Data path: {data_path}")
        
        try:
            # data_basename = os.path.basename(data_path).split('.')[0]
            data_basename = this_json['file_basename'].split('.')[0]
            # Replace spaces and dashes with underscores
            data_basename_proc = data_basename.replace(' ', '_').replace('-', '_')
            this_output_dir = os.path.join(output_base_dir, data_basename_proc)

            if verbose:
                print(f"Data basename: {data_basename}")
                print(f"Processed basename: {data_basename_proc}")
                print(f"Output directory: {this_output_dir}")

            if not os.path.exists(this_output_dir):
                os.mkdir(this_output_dir)
                if verbose:
                    print(f"Created output directory: {this_output_dir}")

            label_mask_path = this_json['fin_mask_path']
            if verbose:
                print(f"Loading label mask from: {label_mask_path}")
            label_mask = np.load(label_mask_path)
            if verbose:
                print(f"Label mask shape: {label_mask.shape}")
            
            # metadata_path = os.path.join(metadata_dir, file_basename + '.csv')
            metadata_path = this_json['wanted_regions_frame_path']
            if verbose:
                print(f"Loading metadata from: {metadata_path}")
            metadata = pd.read_csv(metadata_path)
            if verbose:
                print(f"Metadata shape: {metadata.shape}")

            mask = label_mask.copy()
            if verbose:
                print(f"Opening slide: {data_path}")
            
            # Use slide_handler for consistent slide opening
            slide_metadata = utils.slide_handler(data_path)
            slide = slide_metadata.slide
            scene = slide_metadata.scene
            resolution = scene.resolution[0]  # meters / pixel
            size_meteres = np.array(scene.size) * np.array(resolution)
            down_mag = mask.shape[0] / scene.rect[2]
            mask_resolution = resolution / down_mag

            if verbose:
                print(f"Scene size: {scene.size}")
                print(f"Resolution: {resolution:.2e} meters/pixel")
                print(f"Size in meters: {size_meteres}")
                print(f"Down magnification: {down_mag:.2f}")
                print(f"Mask resolution: {mask_resolution:.2e} meters/pixel")

            # Get tissue dimenions
            # wanted_labels = [1,4]
            wanted_labels = metadata.loc[metadata['tissue_type']
                                         == 'heart']['tissue_num'].values
            if verbose:
                print(f"Heart tissue labels found: {wanted_labels}")
            
            # wanted_mask = mask == wanted_label
            wanted_mask = np.isin(mask, wanted_labels)
            if verbose:
                print(f"Heart tissue pixels: {np.sum(wanted_mask)} / {wanted_mask.size} ({100*np.sum(wanted_mask)/wanted_mask.size:.1f}%)")

        # tissue_props = metadata.loc[metadata['tissue_num'] == wanted_label]
        # major_len = tissue_props['axis_major_length'].values[0]
        # minor_len = tissue_props['axis_minor_length'].values[0]
        # down_mag = mask.shape[0] / scene.rect[2]
        # len_array_raw = np.array([major_len, minor_len])
        # len_array_meters = (len_array_raw / down_mag) * resolution

            ##############################
            if verbose:
                print("Removing mask edges...")
            eroded_mask = remove_mask_edge(
                wanted_mask,
                mask_resolution,
                closing_len=75e-6,
                edge_len=100e-6,
            )
            if verbose:
                print(f"Eroded mask pixels: {np.sum(eroded_mask)} / {eroded_mask.size} ({100*np.sum(eroded_mask)/eroded_mask.size:.1f}%)")

            window_len = 7e-4  # meters
            window_shape_pixels = int(window_len / resolution)

            if verbose:
                print(f"Window length: {window_len:.1e} meters")
                print(f"Window shape in pixels: {window_shape_pixels}")

            window_shape = [window_shape_pixels, window_shape_pixels]
            step_shape = window_shape.copy()

            if verbose:
                print("Generating step windows...")
            step_list = utils.gen_step_windows(
                step_shape, window_shape, scene.rect[2:])
            if verbose:
                print(f"Generated {len(step_list)} step windows")

            if verbose:
                print("Finding wanted sections...")
            _, wanted_sections = utils.get_wanted_sections(
                scene,
                eroded_mask,
                step_list,
                min_fraction=1,
            )
            if verbose:
                print(f"Found {len(wanted_sections)} wanted sections")

            if verbose:
                print("Creating section visualization...")
            fig, ax = utils.visualize_sections(
                scene,
                wanted_sections,
                plot_n=-1,
                edgecolor='orange',
                return_image=True,
            )
            ax.legend().set_visible(False)
            viz_path = os.path.join(
                this_output_dir,
                data_basename_proc + '_' + 'selected_section_visualization.png')
            fig.savefig(
                viz_path,
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(fig)
            if verbose:
                print(f"Saved visualization to: {viz_path}")

            if verbose:
                print("Annotating sections...")
            fin_mask = mask.copy()
            fin_mask[~eroded_mask] = 0
            section_frame = utils.annotate_sections(
                scene,
                fin_mask,
                metadata,
                wanted_sections,
            )
            if verbose:
                print(f"Annotated {len(section_frame)} sections")

            section_labels = section_frame['label_values'].astype(str) + '_' + \
                section_frame['tissue_type']

            section_frame['section_labels'] = section_labels

            if verbose:
                print("Generating section hashes...")
            # Generate truly unique identifiers for each section
            section_frame['section_hash'] = [
                # str(uuid.uuid4().int)[:16]
                str_to_hash(data_basename_proc + '_' + str(section_frame.iloc[i]))
                for i in range(len(section_frame))
            ]

            # Make sure section_bounds are a list (otherwise they are converted weirdly to np.int64)
            # This way they are easier to load
            section_frame['section_bounds'] = section_frame['section_bounds'].apply(
                lambda x: [int(y) for y in x]
            )

            # Write out section_frame
            section_frame_path = os.path.join(
                this_output_dir,
                # file_basename_proc + '_' + 'section_frame.csv'),
                data_basename_proc + '_' + 'section_frame.csv')
            section_frame.to_csv(
                section_frame_path,
                index=False,
            )
            if verbose:
                print(f"Saved section frame to: {section_frame_path}")

            if verbose:
                print("Extracting section images...")
            img_section_list, img_list = utils.output_sections(
                scene,
                section_frame['section_bounds'].to_list(),
                this_output_dir,
                down_sample=4,
                random_output_n=None,
                output_type='return',
            )
            if verbose:
                print(f"Extracted {len(img_list)} section images")

            # Write out images
            out_image_dir = os.path.join(this_output_dir, 'images')
            if not os.path.exists(out_image_dir):
                os.mkdir(out_image_dir)
                if verbose:
                    print(f"Created image output directory: {out_image_dir}")
            
            if verbose:
                print("Writing out images...")
            utils.write_out_images(
                img_list,
                section_frame,
                out_image_dir,
            )
            if verbose:
                print(f"Saved images to: {out_image_dir}")

            # Add path to section frame to json
            this_json['suggested_regions_frame_path'] = os.path.join(
                this_output_dir,
                data_basename_proc + '_' + 'section_frame.csv')

            # Calculate and log processing time
            end_time = time.time()
            processing_time = end_time - start_time
            this_json['suggest_regions_processing_time_seconds'] = processing_time

            with open(json_path, 'w') as f:
                json.dump(this_json, f, indent=4)
            
            if verbose:
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Updated JSON file: {json_path}")
                print(f"Successfully processed {data_basename_proc}")

        except Exception as e:
            # Still log time even if there was an error
            end_time = time.time()
            processing_time = end_time - start_time
            this_json['suggest_regions_processing_time_seconds'] = processing_time
            this_json['suggest_regions_error'] = str(e)
            
            with open(json_path, 'w') as f:
                json.dump(this_json, f, indent=4)
            
            print(f"Error processing {data_basename if 'data_basename' in locals() else 'unknown file'}: {e}")
            print(f"Processing time before error: {processing_time:.2f} seconds")
            if verbose:
                import traceback
                traceback.print_exc()
    
    if verbose:
        print(f"\n{'='*60}")
        print("Region suggestion pipeline completed!")


if __name__ == "__main__":
    main()
