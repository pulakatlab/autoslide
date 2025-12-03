"""
Functions to help with annotation of functions

Preliminary idea being that tissues need to be manually delineated

Steps:
1. Load image
2. Output image
3. Image is manually annotated
4. Image is loaded back in
5. Image is convex hull'd and regions segmented
6. Separately, user is asked to input tissue type
"""

import os
import json
import pylab as plt
import cv2 as cv
import numpy as np
import argparse
from pprint import pprint
from glob import glob
import pandas as pd
from tqdm import tqdm
from skimage.morphology import binary_dilation as dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters.rank import gradient
from scipy.ndimage import binary_fill_holes

# Import config
from src import config
# Import utilities directly
from autoslide.pipeline import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initial annotation of slide images')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output with detailed processing information')
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = args.verbose

    ############################################################
    # PARAMS
    down_sample = 100
    dilation_kern_size = 2
    area_threshold = 10000

    ############################################################

    # Get directories from config
    data_dir = config['data_dir']
    svs_dir = config['svs_dir']
    glob_pattern = '*.svs'
    file_list = glob(os.path.join(svs_dir, '**', glob_pattern), recursive=True)

    if verbose:
        print(f"Configuration loaded:")
        print(f"  Data directory: {data_dir}")
        print(f"  SVS directory: {svs_dir}")
        print(f"  Found {len(file_list)} SVS files to process")
        print(f"Processing parameters:")
        print(f"  Down sample factor: {down_sample}")
        print(f"  Dilation kernel size: {dilation_kern_size}")
        print(f"  Area threshold: {area_threshold}")
        print(f"  {len(file_list)} files found to process")
        pprint(file_list)
        print()

    annot_dir = config['initial_annotation_dir']
    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)
        if verbose:
            print(f"Created annotation directory: {annot_dir}")

    tracking_dir = config['tracking_dir']
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)
        if verbose:
            print(f"Created tracking directory: {tracking_dir}")

    for i, data_path in enumerate(tqdm(file_list)):

        file_basename = os.path.basename(data_path)

        if verbose:
            print(f"Processing file {i+1}/{len(file_list)}: {file_basename}")
            print(f"  Full path: {data_path}")

        slide_handler = utils.slide_handler(data_path)
        scene = slide_handler.scene
        image_rect = np.array(scene.rect) // down_sample
        image = scene.read_block(size=image_rect[2:])

        if verbose:
            print(
                f"  Image loaded - Original size: {scene.rect[2:4]}, Downsampled size: {image_rect[2:4]}")

        ############################################################

        if verbose:
            print(f"  Generating threshold mask...")
        threshold_mask = utils.get_threshold_mask(scene, down_sample=down_sample)

        if verbose:
            print(f"  Applying morphological dilation...")
        dilation_kern = np.ones((dilation_kern_size, dilation_kern_size))
        dilated_mask = dilation(threshold_mask, footprint=dilation_kern)

        if verbose:
            print(f"  Labeling connected components...")
        label_image = label(dilated_mask)
        regions = regionprops(label_image)
        image_label_overlay = label2rgb(
            label_image, image=dilated_mask, bg_label=0)

        if verbose:
            print(f"  Found {len(regions)} regions total")

        wanted_feature_names = [
            'label',
            'area',
            'eccentricity',
            'axis_major_length',
            'axis_minor_length',
            'eccentricity',
            'solidity',
            'centroid',
        ]

        if verbose:
            print(
                f"  Extracting region features: {', '.join(wanted_feature_names)}")

        wanted_features = [
            [getattr(region, feature_name) for region in regions]
            for feature_name in wanted_feature_names
        ]

        region_frame = pd.DataFrame(
            {feature_name: feature for feature_name, feature in
             zip(wanted_feature_names, wanted_features)}
        )

        region_frame.sort_values('label', ascending=True, inplace=True)
        wanted_regions_frame = region_frame[region_frame.area > area_threshold]
        wanted_regions = wanted_regions_frame.label.values

        if verbose:
            print(
                f"  Filtered to {len(wanted_regions)} regions above area threshold ({area_threshold})")
            if len(wanted_regions) > 0:
                print(
                    f"  Region areas: min={wanted_regions_frame.area.min():.0f}, max={wanted_regions_frame.area.max():.0f}, mean={wanted_regions_frame.area.mean():.0f}")

        # Output wanted_regions_frame to be annotated manually
        wanted_regions_frame['tissue_type'] = np.nan
        wanted_regions_frame['tissue_num'] = np.nan

        csv_path = os.path.join(
            annot_dir, file_basename.replace('.svs', '.csv'))
        wanted_regions_frame.to_csv(csv_path, index=False)

        if verbose:
            print(f"  Saved region metadata to: {csv_path}")

        # Drop regions that are not wanted
        fin_label_image = label_image.copy()
        dropped_count = 0
        for i in region_frame.label.values:
            if i not in wanted_regions:
                fin_label_image[fin_label_image == i] = 0
                dropped_count += 1

        if verbose:
            print(f"  Dropped {dropped_count} regions below area threshold")

        # Write out image with regions labelled
        npy_path = os.path.join(
            annot_dir, file_basename.replace('.svs', '.npy'))
        np.save(npy_path, fin_label_image)

        if verbose:
            print(f"  Saved label image to: {npy_path}")

        if verbose:
            print(f"  Generating visualization plots...")

        image_label_overlay = label2rgb(fin_label_image,
                                        image=fin_label_image > 0,
                                        bg_label=0)

        filled_binary = binary_fill_holes(fin_label_image > 0)*1

        gradient_image = gradient(filled_binary, np.ones((10, 10)))
        grad_inds = np.where(gradient_image > 0)

        if verbose:
            print(f"  Creating 5-panel visualization plot...")

        fig, ax = plt.subplots(1, 5,
                               sharex=True, sharey=True,
                               figsize=(20, 10)
                               )
        ax[0].imshow(np.swapaxes(image, 0, 1))
        ax[1].imshow(threshold_mask)
        ax[2].imshow(dilated_mask)
        ax[3].imshow(image_label_overlay)
        ax[4].imshow(np.swapaxes(image, 0, 1))
        ax[4].scatter(grad_inds[1], grad_inds[0],
                      s=1, color='orange', alpha=0.7,
                      label='Outline')
        ax[0].set_title('Original')
        ax[1].set_title('Threshold')
        ax[2].set_title('Dilated')
        ax[3].set_title('Labelled')
        ax[4].set_title('Outline Overlay')
        ax[4].legend()
        # Add labels at the center of each region
        for this_row in wanted_regions_frame.itertuples():
            ax[3].text(this_row.centroid[1], this_row.centroid[0],
                       this_row.label, color='r', fontsize=25,
                       weight='bold')
        fig.suptitle(file_basename)
        plt.tight_layout()

        plot_path = os.path.join(
            annot_dir, file_basename.replace('.svs', '.png'))
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

        if verbose:
            print(f"  Saved visualization plot to: {plot_path}")

        # Write out a json with:
        # - file_basename
        # - data_path
        # - fin_label_image path
        # - wanted_regions_frame path

        json_data = {
            'file_basename': file_basename,
            'data_path': data_path,
            'initial_mask_path': os.path.join(annot_dir, file_basename.replace('.svs', '.npy')),
            'wanted_regions_frame_path': os.path.join(annot_dir, file_basename.replace('.svs', '.csv')),
        }

        json_path = os.path.join(
            tracking_dir, file_basename.replace('.svs', '.json'))
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

        if verbose:
            print(f"  Saved tracking JSON to: {json_path}")
            print(f"  Completed processing {file_basename}")
            print()

    if verbose:
        print(f"Initial annotation processing complete!")
        print(f"Processed {len(file_list)} files total")
        print(f"Output files saved to:")
        print(f"  Annotations: {annot_dir}")
        print(f"  Tracking: {tracking_dir}")


if __name__ == "__main__":
    main()
