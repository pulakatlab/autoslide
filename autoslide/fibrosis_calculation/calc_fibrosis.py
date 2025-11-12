"""
Functionality to calculate fibrosis given a single image + optional mask
"""

import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import json
import time
from tqdm import tqdm
from ast import literal_eval
from autoslide import config as autoslide_config
from autoslide.pipeline import utils
from autoslide.utils.get_section_from_hash import *
import cv2
from joblib import Parallel, delayed

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

    # OpenCV expected images in BGR format to have range [0, 255]
    if image.dtype != np.uint8:
        scaled_image = (image * 255).astype(np.uint8)
    else:
        scaled_image = image

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2HSV)

    # Extract hue channel
    hue_channel = hsv_image[:, :, 0]

    # Normalize hue channel to [0, 1] range
    hue_channel = hue_channel / 180.0  # OpenCV uses [0, 180] for hue

    # Create a mask based on the hue value and width from the config
    mask = ((hue_channel >= config['hue_value'] - config['hue_width'] / 2) &
            (hue_channel <= config['hue_value'] + config['hue_width'] / 2))

    # figure, ax = plt.subplots(1, 3, figsize=(15, 5),
    #                           sharex=True, sharey=True)
    # ax[0].imshow(image)
    # ax[0].set_title('Original Image')
    # ax[1].imshow(hsv_image)
    # ax[1].set_title('HSV Image')
    # ax[2].imshow(mask, cmap='gray')
    # ax[2].set_title('Fibrosis Mask')
    # plt.show()

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
        - vessel_mask: Optional mask for blood vessels to exclude from analysis.

    Returns:
        - A dictionary containing the fibrosis area, total area, and fibrosis percentage.

    """
    if mask is None:
        mask = gen_fibrosis_mask(image, config, vessel_mask=vessel_mask)

    # Calculate areas
    fibrosis_area = np.sum(mask)
    total_area = image.shape[0] * image.shape[1]

    # If vessel mask is provided, exclude vessel areas from total area
    if vessel_mask is not None:
        # Convert vessel mask to binary if needed
        if vessel_mask.max() > 1:
            vessel_binary = (vessel_mask > 127).astype(np.uint8)
        else:
            vessel_binary = vessel_mask.astype(np.uint8)

        vessel_area = np.sum(vessel_binary)
        tissue_area = total_area - vessel_area

        # Also exclude vessels from fibrosis area
        fibrosis_mask_no_vessels = mask & (~vessel_binary.astype(bool))
        fibrosis_area_no_vessels = np.sum(fibrosis_mask_no_vessels)

        # Calculate fibrotic pixels within vessel areas
        fibrosis_in_vessels = mask & vessel_binary.astype(bool)
        fibrosis_in_vessels_area = np.sum(fibrosis_in_vessels)

        fibrosis_percentage = (fibrosis_area_no_vessels /
                               tissue_area) * 100 if tissue_area > 0 else 0

        return {
            'fibrosis_area': int(fibrosis_area),
            'fibrosis_area_no_vessels': int(fibrosis_area_no_vessels),
            'fibrosis_in_vessels_area': int(fibrosis_in_vessels_area),
            'vessel_area': int(vessel_area),
            'tissue_area': int(tissue_area),
            'total_area': int(total_area),
            'fibrosis_percentage': float(fibrosis_percentage),
            'fibrosis_percentage_total': float((fibrosis_area / total_area) * 100)
        }
    else:
        fibrosis_percentage = (fibrosis_area / total_area) * 100

        return {
            'fibrosis_area': int(fibrosis_area),
            'total_area': int(total_area),
            'fibrosis_percentage': float(fibrosis_percentage)
        }

##############################


def find_images_with_masks(data_dir, verbose=False):
    """
    Find all images in suggested_regions that have corresponding neural network predicted masks.

    Returns:
        list: List of dictionaries containing image paths, mask paths, and metadata
    """
    suggested_regions_dir = os.path.join(data_dir, 'suggested_regions')

    if not os.path.exists(suggested_regions_dir):
        print(
            f"Suggested regions directory not found: {suggested_regions_dir}")
        return []

    # Find all image files
    image_pattern = os.path.join(
        suggested_regions_dir, '**', 'images', '*.png')
    image_paths = glob(image_pattern, recursive=True)

    images_with_masks = []

    for image_path in image_paths:
        # Determine corresponding mask path
        mask_path = image_path.replace('/images/', '/masks/')
        mask_path = mask_path.replace('.png', '_mask.png')

        # Check if mask exists
        if os.path.exists(mask_path):
            # Extract metadata from path
            path_parts = image_path.split(os.sep)
            region_type = None
            for part in path_parts:
                if 'sections' in part:
                    region_type = part
                    break

            image_name = os.path.basename(image_path)
            hash_value = image_name.split(
                '_')[-1].replace('.png', '') if '_' in image_name else None

            images_with_masks.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'image_name': image_name,
                'region_type': region_type,
                'hash_value': hash_value
            })

    if verbose:
        print(f"Found {len(images_with_masks)} images with corresponding masks")
        if images_with_masks:
            print("Sample entries:")
            for i, item in enumerate(images_with_masks[:3]):
                print(
                    f"  {i+1}. {item['image_name']} (type: {item['region_type']})")

    return images_with_masks


def process_single_image_fibrosis(
        item,
        fibrosis_config,
        save_visualizations,
        vis_dir,
        results_dir,
        verbose
):
    """
    Process a single image for fibrosis quantification.

    Args:
        item (dict): Dictionary containing image metadata and paths
        fibrosis_config (dict): Configuration for fibrosis detection
        save_visualizations (bool): Whether to save visualization images
        vis_dir (str): Directory to save visualizations
        results_dir (str): Directory to save individual CSV results
        verbose (bool): Whether to print detailed information

    Returns:
        tuple: (success_flag, result_entry_or_error_message)
    """
    image_path = item['image_path']
    mask_path = item['mask_path']
    image_name = item['image_name']

    try:
        # Load image and vessel mask
        image = plt.imread(image_path)
        if image.shape[-1] == 4:  # Remove alpha channel if present
            image = image[..., :3]

        vessel_mask = plt.imread(mask_path)
        if len(vessel_mask.shape) == 3:  # Convert to grayscale if needed
            vessel_mask = vessel_mask[..., 0]

        if verbose:
            print(f"  Processing {image_name}")
            print(f"  Image shape: {image.shape}")
            print(f"  Vessel mask shape: {vessel_mask.shape}")
            print(
                f"  Vessel mask range: {vessel_mask.min()} - {vessel_mask.max()}")

        # Generate fibrosis mask
        fibrosis_mask = gen_fibrosis_mask(image, fibrosis_config)

        # Quantify fibrosis
        fibrosis_results = quantify_fibrosis(
            image,
            mask=fibrosis_mask,
            config=fibrosis_config,
            vessel_mask=vessel_mask
        )

        # Add metadata to results
        result_entry = {
            'image_name': image_name,
            'image_path': image_path,
            'mask_path': mask_path,
            'region_type': item['region_type'],
            'hash_value': item['hash_value'],
            **fibrosis_results
        }

        if verbose:
            if 'fibrosis_percentage' in fibrosis_results:
                print(
                    f"  Fibrosis percentage: {fibrosis_results['fibrosis_percentage']:.2f}%")
            if 'vessel_area' in fibrosis_results:
                print(
                    f"  Vessel area: {fibrosis_results['vessel_area']} pixels")

        # Save individual CSV result
        csv_filename = image_name.replace('.png', '_fibrosis_results.csv')
        csv_path = os.path.join(
            results_dir, 'individual_results', csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        result_df = pd.DataFrame([result_entry])
        result_df.to_csv(csv_path, index=False)

        if verbose:
            print(f"  Individual results saved to: {csv_path}")

        # Create visualization if requested
        if save_visualizations and vis_dir:
            create_fibrosis_visualization(
                image, fibrosis_mask, vessel_mask, fibrosis_results,
                image_name, vis_dir, verbose=verbose
            )

        return True, result_entry

    except Exception as e:
        error_msg = f"Error processing {image_name}: {e}"
        if verbose:
            import traceback
            error_msg += f"\n  Full traceback: {traceback.format_exc()}"
        return False, error_msg


def check_existing_outputs(images_with_masks, results_dir, save_visualizations, verbose=False):
    """
    Check which images already have outputs and filter out processed ones.

    Args:
        images_with_masks (list): List of image dictionaries
        results_dir (str): Results directory path
        save_visualizations (bool): Whether visualizations are being saved
        verbose (bool): Whether to print detailed information

    Returns:
        list: Filtered list of images that need processing
    """
    # Check for existing individual CSV results
    individual_results_dir = os.path.join(results_dir, 'individual_results')

    # Filter out images that already have results
    images_to_process = []
    skipped_count = 0

    for item in images_with_masks:
        image_name = item['image_name']

        # Check if individual CSV result exists
        csv_filename = image_name.replace('.png', '_fibrosis_results.csv')
        csv_path = os.path.join(individual_results_dir, csv_filename)
        has_csv_result = os.path.exists(csv_path)

        # Check if visualization exists (if visualizations are being saved)
        has_visualization = True  # Default to true if not saving visualizations
        if save_visualizations:
            vis_dir = os.path.join(results_dir, 'visualizations')
            vis_filename = image_name.replace('.png', '_fibrosis_analysis.png')
            vis_path = os.path.join(vis_dir, vis_filename)
            has_visualization = os.path.exists(vis_path)

        # Skip if both outputs exist
        if has_csv_result and has_visualization:
            skipped_count += 1
            if verbose:
                print(f"Skipping {image_name} - outputs already exist")
        else:
            images_to_process.append(item)
            if verbose and has_csv_result and not has_visualization:
                print(f"Will reprocess {image_name} - missing visualization")
            elif verbose and not has_csv_result and has_visualization:
                print(f"Will reprocess {image_name} - missing CSV result")

    if verbose:
        print(f"Skipped {skipped_count} images with existing outputs")
        print(f"Will process {len(images_to_process)} images")

    return images_to_process


def process_all_fibrosis_quantification(
        data_dir=None,
        fibrosis_config=None,
        save_visualizations=True,
        max_images=None,
        verbose=False,
        n_jobs=-1,
        force_run=False
):
    """
    Process all images with neural network masks for fibrosis quantification.

    Args:
        data_dir (str): Data directory path
        fibrosis_config (dict): Configuration for fibrosis detection
        save_visualizations (bool): Whether to save visualization images
        max_images (int): Maximum number of images to process (for testing)
        verbose (bool): Whether to print detailed information
        n_jobs (int): Number of parallel jobs (-1 for all available cores)
        force_run (bool): If True, process all images regardless of existing outputs
    """
    if data_dir is None:
        data_dir = config['data_dir']

    if fibrosis_config is None:
        # Default fibrosis configuration
        # Values taken from: https://pmc.ncbi.nlm.nih.gov/articles/PMC5376943/
        fibrosis_config = {
            'hue_value': 0.6785,
            'hue_width': 0.4,
            'color_saturation_threshold': 0.0
        }

    print("Starting fibrosis quantification on all images with neural network masks...")

    if verbose:
        print(f"Configuration:")
        print(f"  Data directory: {data_dir}")
        print(f"  Fibrosis config: {fibrosis_config}")
        print(f"  Save visualizations: {save_visualizations}")
        print(f"  Max images: {max_images if max_images else 'unlimited'}")
        print(
            f"  Parallel jobs: {n_jobs if n_jobs != -1 else 'all available cores'}")
        print(f"  Force run: {force_run}")

    # Find images with masks
    images_with_masks = find_images_with_masks(data_dir, verbose=verbose)

    if not images_with_masks:
        print("No images found with corresponding neural network masks.")
        return

    # Create output directories
    results_dir = os.path.join(data_dir, 'fibrosis_results')
    os.makedirs(results_dir, exist_ok=True)

    vis_dir = None
    if save_visualizations:
        vis_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    # Filter out images that already have outputs (unless force_run is True)
    if not force_run:
        images_with_masks = check_existing_outputs(
            images_with_masks, results_dir, save_visualizations, verbose=verbose
        )

        if not images_with_masks:
            print(
                "All images already have outputs. Use --force-run to reprocess all images.")
            return

    # Limit number of images if specified
    if max_images and max_images < len(images_with_masks):
        images_with_masks = images_with_masks[:max_images]
        print(f"Limited processing to {max_images} images for testing")

    # Group images by SVS directory for timing tracking
    svs_groups = {}
    for item in images_with_masks:
        # Extract SVS directory from image path
        # Path structure: data_dir/suggested_regions/svs_name/region_type/images/image.png
        path_parts = item['image_path'].split(os.sep)
        suggested_regions_idx = None
        for i, part in enumerate(path_parts):
            if part == 'suggested_regions':
                suggested_regions_idx = i
                break

        if suggested_regions_idx is not None and suggested_regions_idx + 1 < len(path_parts):
            svs_name = path_parts[suggested_regions_idx + 1]
            if svs_name not in svs_groups:
                svs_groups[svs_name] = []
            svs_groups[svs_name].append(item)

    if verbose:
        print(f"Found {len(svs_groups)} SVS directories to process:")
        for svs_name, items in svs_groups.items():
            print(f"  {svs_name}: {len(items)} images")

    # Process each SVS group and track timing
    all_results = []
    total_successful = 0
    total_failed = 0

    for svs_name, svs_images in tqdm(svs_groups.items()):
        print(f"\nProcessing SVS: {svs_name} ({len(svs_images)} images)")
        svs_start_time = time.time()

        # Process images in parallel for this SVS
        results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
            delayed(process_single_image_fibrosis)(
                item, fibrosis_config, save_visualizations, vis_dir, results_dir, verbose
            ) for item in tqdm(svs_images, desc=f"Processing {svs_name}")
        )

        # Calculate timing for this SVS
        svs_end_time = time.time()
        svs_processing_time = svs_end_time - svs_start_time

        # Count successful/failed for this SVS
        svs_successful = 0
        svs_failed = 0
        for success, result_or_error in results:
            if success:
                all_results.append(result_or_error)
                svs_successful += 1
                total_successful += 1
            else:
                print(result_or_error)
                svs_failed += 1
                total_failed += 1

        # Log timing to tracking JSON for this SVS
        svs_dir_path = os.path.join(data_dir, 'suggested_regions', svs_name)
        save_svs_timing_to_tracking_json(
            svs_dir_path,
            svs_processing_time,
            len(svs_images),
            verbose=verbose
        )

        print(f"SVS {svs_name} completed in {svs_processing_time:.2f} seconds")
        print(f"  Successful: {svs_successful}, Failed: {svs_failed}")

    # Create combined CSV from all individual results
    if all_results or not force_run:
        # Load all individual CSV files to create combined results
        individual_results_dir = os.path.join(
            results_dir, 'individual_results')
        if os.path.exists(individual_results_dir):
            individual_csv_files = glob(os.path.join(
                individual_results_dir, '*_fibrosis_results.csv'))

            if individual_csv_files:
                # Read all individual CSV files
                combined_results = []
                for csv_file in individual_csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        combined_results.append(df)
                    except Exception as e:
                        if verbose:
                            print(f"Error reading {csv_file}: {e}")

                if combined_results:
                    # Combine all results
                    combined_df = pd.concat(
                        combined_results, ignore_index=True)

                    # Save combined CSV
                    csv_path = os.path.join(
                        results_dir, 'fibrosis_quantification_results.csv')
                    combined_df.to_csv(csv_path, index=False)
                    print(f"Combined results saved to: {csv_path}")
                    print(f"Total results: {len(combined_df)} images")

                    # Create summary statistics
                    create_summary_statistics(
                        combined_df, results_dir, verbose=verbose)
                else:
                    print("No valid individual CSV files found to combine")
            else:
                print("No individual CSV files found")
        else:
            print("Individual results directory not found")

    print(f"\nFibrosis quantification complete!")
    print(f"Successful analyses: {total_successful}")
    print(f"Failed analyses: {total_failed}")
    print(f"Total processed: {total_successful + total_failed}")


def create_fibrosis_visualization(image, fibrosis_mask, vessel_mask, results, image_name, vis_dir, verbose=False):
    """
    Create and save a visualization of fibrosis analysis results.
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Vessel mask
        axes[0, 1].imshow(vessel_mask, cmap='gray')
        axes[0, 1].set_title('Vessel Mask (Neural Network)')
        axes[0, 1].axis('off')

        # Fibrosis mask
        axes[0, 2].imshow(fibrosis_mask, cmap='gray')
        axes[0, 2].set_title('Fibrosis Mask')
        axes[0, 2].axis('off')

        # Fibrosis overlay
        fib_overlay = image.copy()
        fib_overlay[fibrosis_mask == 1] = [0, 1, 0]  # Green for fibrosis
        axes[1, 0].imshow(fib_overlay)
        axes[1, 0].set_title('Fibrosis Overlay (Green)')
        axes[1, 0].axis('off')

        # Combined overlay
        combined_overlay = image.copy()
        vessel_binary = (vessel_mask > 127).astype(
            bool) if vessel_mask.max() > 1 else vessel_mask.astype(bool)
        combined_overlay[vessel_binary] = [1, 0, 0]  # Red for vessels
        combined_overlay[fibrosis_mask == 1] = [0, 1, 0]  # Green for fibrosis
        axes[1, 1].imshow(combined_overlay)
        axes[1, 1].set_title('Combined Overlay\n(Red=Vessels, Green=Fibrosis)')
        axes[1, 1].axis('off')

        # Results text
        axes[1, 2].axis('off')
        result_text = f"Fibrosis Analysis Results\n\n"
        result_text += f"Image: {image_name}\n\n"

        if 'fibrosis_percentage' in results:
            result_text += f"Fibrosis %: {results['fibrosis_percentage']:.2f}%\n"
        if 'fibrosis_area' in results:
            result_text += f"Fibrosis Area: {results['fibrosis_area']:,} px\n"
        if 'fibrosis_in_vessels_area' in results:
            result_text += f"Fibrosis in Vessels: {results['fibrosis_in_vessels_area']:,} px\n"
        if 'vessel_area' in results:
            result_text += f"Vessel Area: {results['vessel_area']:,} px\n"
        if 'tissue_area' in results:
            result_text += f"Tissue Area: {results['tissue_area']:,} px\n"
        if 'total_area' in results:
            result_text += f"Total Area: {results['total_area']:,} px\n"

        axes[1, 2].text(0.1, 0.9, result_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        # Save visualization
        vis_filename = image_name.replace('.png', '_fibrosis_analysis.png')
        vis_path = os.path.join(vis_dir, vis_filename)
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if verbose:
            print(f"  Visualization saved to: {vis_path}")

    except Exception as e:
        print(f"Error creating visualization for {image_name}: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")


def save_svs_timing_to_tracking_json(svs_dir_path, svs_processing_time, num_images, verbose=False):
    """
    Save SVS processing timing information to the tracking JSON file.

    Args:
        svs_dir_path (str): Path to the SVS directory
        svs_processing_time (float): Time taken to process the SVS in seconds
        num_images (int): Number of images processed for this SVS
        verbose (bool): Whether to print detailed information
    """
    try:
        # Find the tracking JSON file - look in the data_dir/tracking directory
        # based on the SVS directory name
        svs_name = os.path.basename(svs_dir_path).split('.')[0]
        data_dir = autoslide_config['data_dir']
        tracking_dir = os.path.join(data_dir, 'tracking')

        # Look for tracking files that match this SVS
        tracking_file = os.path.join(tracking_dir, f"{svs_name}.json")

        if not tracking_file:
            if verbose:
                print(f"No tracking JSON file found for SVS: {svs_name}")
                print(f"Looking for filepath: {tracking_file}")
            return

        # Load existing tracking data
        try:
            with open(tracking_file, 'r') as f:
                tracking_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            tracking_data = {}

        # Add fibrosis processing timing (following the pattern from final_annotation.py)
        tracking_data['fibrosis_processing_time'] = svs_processing_time
        tracking_data['fibrosis_num_images_processed'] = num_images
        tracking_data['fibrosis_processing_time_per_image'] = svs_processing_time / \
            num_images if num_images > 0 else 0
        tracking_data['fibrosis_processing_timestamp'] = time.strftime(
            '%Y-%m-%d %H:%M:%S')

        # Save updated tracking data
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=4)

        if verbose:
            print(f"Fibrosis timing saved to {tracking_file}")
            print(f"  Processing time: {svs_processing_time:.2f} seconds")
            print(f"  Images processed: {num_images}")
            print(
                f"  Time per image: {svs_processing_time / num_images:.2f} seconds" if num_images > 0 else "  Time per image: N/A")

    except Exception as e:
        print(f"Error saving timing to tracking JSON: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")


def create_summary_statistics(results_df, results_dir, verbose=False):
    """
    Create and save summary statistics and plots.
    """
    try:
        # Summary statistics
        summary_stats = {
            'total_images': len(results_df),
            'mean_fibrosis_percentage': results_df['fibrosis_percentage'].mean(),
            'median_fibrosis_percentage': results_df['fibrosis_percentage'].median(),
            'std_fibrosis_percentage': results_df['fibrosis_percentage'].std(),
            'min_fibrosis_percentage': results_df['fibrosis_percentage'].min(),
            'max_fibrosis_percentage': results_df['fibrosis_percentage'].max()
        }

        # Save summary statistics
        summary_path = os.path.join(results_dir, 'summary_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        if verbose:
            print(f"Summary statistics saved to: {summary_path}")
            print(
                f"Mean fibrosis percentage: {summary_stats['mean_fibrosis_percentage']:.2f}%")

        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Histogram of fibrosis percentages
        axes[0, 0].hist(results_df['fibrosis_percentage'],
                        bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Fibrosis Percentage (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Fibrosis Percentages')
        axes[0, 0].axvline(summary_stats['mean_fibrosis_percentage'],
                           color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()

        # Box plot by region type if available
        if 'region_type' in results_df.columns and results_df['region_type'].nunique() > 1:
            region_data = [results_df[results_df['region_type'] == rt]['fibrosis_percentage'].values
                           for rt in results_df['region_type'].unique() if pd.notna(rt)]
            region_labels = [
                rt for rt in results_df['region_type'].unique() if pd.notna(rt)]
            axes[0, 1].boxplot(region_data, labels=region_labels)
            axes[0, 1].set_ylabel('Fibrosis Percentage (%)')
            axes[0, 1].set_title('Fibrosis by Region Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No region type data\navailable',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Fibrosis by Region Type')

        # Scatter plot: fibrosis area vs total area
        axes[1, 0].scatter(results_df['total_area'],
                           results_df['fibrosis_area'], alpha=0.6)
        axes[1, 0].set_xlabel('Total Area (pixels)')
        axes[1, 0].set_ylabel('Fibrosis Area (pixels)')
        axes[1, 0].set_title('Fibrosis Area vs Total Area')

        # Summary statistics text
        axes[1, 1].axis('off')
        stats_text = f"Summary Statistics\n\n"
        stats_text += f"Total Images: {summary_stats['total_images']}\n\n"
        stats_text += f"Fibrosis Percentage:\n"
        stats_text += f"  Mean: {summary_stats['mean_fibrosis_percentage']:.2f}%\n"
        stats_text += f"  Median: {summary_stats['median_fibrosis_percentage']:.2f}%\n"
        stats_text += f"  Std Dev: {summary_stats['std_fibrosis_percentage']:.2f}%\n"
        stats_text += f"  Min: {summary_stats['min_fibrosis_percentage']:.2f}%\n"
        stats_text += f"  Max: {summary_stats['max_fibrosis_percentage']:.2f}%\n"

        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        # Save summary plot
        summary_plot_path = os.path.join(results_dir, 'summary_plots.png')
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if verbose:
            print(f"Summary plots saved to: {summary_plot_path}")

    except Exception as e:
        print(f"Error creating summary statistics: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Fibrosis quantification on images with neural network masks')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory path (default: from config)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Skip saving visualization images')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information during processing')
    parser.add_argument('--hue-value', type=float, default=0.6785,
                        help='Hue value for fibrosis detection (default: 0.6785)')
    parser.add_argument('--hue-width', type=float, default=0.4,
                        help='Hue width for fibrosis detection (default: 0.4)')
    parser.add_argument('--saturation-threshold', type=float, default=0.0,
                        help='Color saturation threshold (default: 0.0)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all available cores, default: -1)')
    parser.add_argument('--force-run', action='store_true',
                        help='Process all images regardless of existing outputs')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create fibrosis configuration from arguments
    fibrosis_config = {
        'hue_value': args.hue_value,
        'hue_width': args.hue_width,
        'color_saturation_threshold': args.saturation_threshold
    }

    process_all_fibrosis_quantification(
        data_dir=args.data_dir,
        fibrosis_config=fibrosis_config,
        save_visualizations=not args.no_visualizations,
        max_images=args.max_images,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        force_run=args.force_run
    )
