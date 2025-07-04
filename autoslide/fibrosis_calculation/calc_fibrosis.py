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
import cv2

##############################


def gen_fibrosis_mask(
        image,
        config,
        vessel_mask=None,
):
    """
    Generate a mask for fibrosis in the image based on the provided configuration.

    Parameters:
    - image: The input image to analyze.
    - config: Configuration dictionary containing parameters for fibrosis detection.
    - vessel_mask: Optional mask for blood vessels, not used in this example.

    Returns:
    - mask: A binary mask where fibrosis is detected.
    """

    # Not implemented error for vessel_mask
    if vessel_mask is not None:
        raise NotImplementedError(
            "Vessel mask functionality is not implemented yet.")

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract hue channel
    hue_channel = hsv_image[:, :, 0]

    # Normalize hue channel to [0, 1] range
    hue_channel = hue_channel / 180.0  # OpenCV uses [0, 180] for hue

    # Create a mask based on the hue value and width from the config
    mask = ((hue_channel >= config['hue_value'] - config['hue_width'] / 2) &
            (hue_channel <= config['hue_value'] + config['hue_width'] / 2))

    # Apply color saturation threshold if specified
    if 'color_saturation_threshold' in config:
        saturation_channel = hsv_image[:, :, 1]
        # Normalize saturation channel to [0, 1] range
        saturation_channel = saturation_channel / 255.0
        mask &= (saturation_channel >= config['color_saturation_threshold'])

    return mask.astype(np.uint8)


def quantify_fibrosis(
        image,
        mask=None,
        config=None,
        vessel_mask=None,
):
    """
    Quantify fibrosis in an image using a mask if provided.

    Parameters:
        - image: The input image to analyze.
        - mask: Optional binary mask where fibrosis is detected.
        - config: Configuration dictionary containing parameters for fibrosis detection.
        - vessel_mask: Optional mask for blood vessels, not used in this example.

    Returns:
        - A dictionary containing the fibrosis area, total area, and fibrosis percentage.

    """
    # Not implemented error for vessel_mask
    if vessel_mask is not None:
        raise NotImplementedError(
            "Vessel mask functionality is not implemented yet.")

    if mask is None:
        mask = gen_fibrosis_mask(image, config,)  # vessel_mask=vessel_mask)

    # Placeholder for actual fibrosis quantification logic
    fibrosis_area = np.sum(mask)  # Example: count pixels in the mask
    total_area = image.shape[0] * image.shape[1]

    fibrosis_percentage = (fibrosis_area / total_area) * 100

    return {
        'fibrosis_area': int(fibrosis_area),
        'total_area': total_area,
        'fibrosis_percentage': float(fibrosis_percentage)
    }

##############################


if __name__ == "__main__":

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

    section, section_details = get_section_from_hash(
        test_hash, final_df)

    # if section is not None:
    #     # Visualize the section
    #     # fig, ax = visualize_section(section, utils)
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    #     ax.imshow(section)
    #     fig.suptitle(f"Section from SVS\nSection: {section['basename']}, Hash: {section['section_hash']}" +
    #                  f"\nData Path: {section['data_path']},\nSection Bounds: {section['section_bounds']}")
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print(f"No section found with hash {test_hash}")

    # Example usage of fibrosis quantification
    fibrosis_config = dict(
        hue_value=0.6785,
        hue_width=0.4,
        color_saturation_threshold=0.0
    )

    fibrosis_mask = gen_fibrosis_mask(section, fibrosis_config)
    # Get quantification results
    fibrosis_results = quantify_fibrosis(
        section, mask=fibrosis_mask, config=fibrosis_config)

    fib_img = section.copy()
    # Apply the mask to the original image
    fib_img[~mask] = 255  # Set non-fibrosis areas to black

    fig, ax = plt.subplots(1, 3, figsize=(10, 5),
                           sharex=True, sharey=True, dpi=100)
    ax[0].imshow(section,)
    ax[0].set_title('Original Image')
    ax[1].imshow(fibrosis_mask, cmap='gray')
    ax[1].set_title('Fibrosis Mask')
    ax[2].imshow(fib_img)
    ax[2].set_title('Fibrosis Highlighted')
    fig.suptitle(f"Fibrosis Quantification Results\n"
                 f"Fibrosis Area: {fibrosis_results['fibrosis_area']} pixels\n"
                 f"Total Area: {fibrosis_results['total_area']} pixels\n"
                 f"Fibrosis Percentage: {fibrosis_results['fibrosis_percentage']:.2f}%")
    plt.tight_layout()
    plt.show()
