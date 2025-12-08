"""
Batch prediction script for vessel segmentation on all images in suggested_regions.

This script:
1. Finds all images under data_dir/suggested_regions/*/images/*.png
2. Checks if corresponding masks already exist
3. Performs prediction only on images without existing masks
4. Saves predicted masks to corresponding mask directories
"""

from autoslide.src.pipeline import utils
import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import cv2
import time
import json
import shutil

from autoslide.src import config
from autoslide.src.pipeline.model.prediction_utils import load_model, predict_single_image

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
plot_dir = config['plot_dirs']


##############################

def remove_svs_outputs(svs_dir_path, verbose=False):
    """
    Remove mask and overlay directories for a given SVS directory.

    Args:
        svs_dir_path (str): Path to the SVS directory
        verbose (bool): Whether to print detailed information
    """
    mask_dir = os.path.join(svs_dir_path, 'masks')
    overlay_dir = os.path.join(svs_dir_path, 'overlays')

    dirs_removed = 0

    if os.path.exists(mask_dir):
        if verbose:
            print(f"  Removing mask directory: {mask_dir}")
        shutil.rmtree(mask_dir)
        dirs_removed += 1

    if os.path.exists(overlay_dir):
        if verbose:
            print(f"  Removing overlay directory: {overlay_dir}")
        shutil.rmtree(overlay_dir)
        dirs_removed += 1

    if verbose and dirs_removed > 0:
        print(f"  Removed {dirs_removed} output directories")
    elif verbose:
        print(f"  No output directories found to remove")


def find_images_to_process(reprocess=False, verbose=False, custom_dir=None):
    """
    Find all images that need prediction and don't already have masks.
    Groups images by their parent SVS directory.

    Args:
        reprocess (bool): If True, process all images regardless of existing masks
        verbose (bool): Whether to print detailed information
        custom_dir (str): Custom directory to search for images. If provided, uses this instead of config dir

    Returns:
        dict: Dictionary mapping SVS directory names to lists of image dictionaries
    """
    # Use custom directory if provided, otherwise use config directory
    if custom_dir:
        suggested_regions_dir = custom_dir
        if verbose:
            print(f"Using custom directory: {suggested_regions_dir}")
    else:
        suggested_regions_dir = config['suggested_regions_dir']
        if verbose:
            print(f"Using config directory: {suggested_regions_dir}")

    if verbose:
        print(f"Looking for images in: {suggested_regions_dir}")

    if not os.path.exists(suggested_regions_dir):
        print(
            f"Directory not found: {suggested_regions_dir}")
        return {}

    # Find all image files
    # Support both structured (with 'images' subdirectory) and flat directory structures
    image_pattern_structured = os.path.join(
        suggested_regions_dir, '**', 'images', '*.png')
    image_pattern_flat = os.path.join(
        suggested_regions_dir, '*.png')

    image_paths_structured = glob(image_pattern_structured, recursive=True)
    image_paths_flat = glob(image_pattern_flat, recursive=False)

    # Use structured if found, otherwise use flat
    if image_paths_structured:
        image_paths = image_paths_structured
        is_flat_structure = False
    else:
        image_paths = image_paths_flat
        is_flat_structure = True

    if verbose:
        if is_flat_structure:
            print(f"Search pattern (flat): {image_pattern_flat}")
        else:
            print(f"Search pattern (structured): {image_pattern_structured}")
        print(f"Found {len(image_paths)} images")
        if image_paths:
            print("Sample image paths:")
            for i, path in enumerate(image_paths[:3]):
                print(f"  {i+1}. {path}")
            if len(image_paths) > 3:
                print(f"  ... and {len(image_paths) - 3} more")
    else:
        print(f"Found {len(image_paths)} images")

    # Group images by SVS directory
    images_by_svs = {}
    total_to_process = 0

    for image_path in image_paths:
        if is_flat_structure:
            # For flat structure, use the directory name as SVS name
            svs_dir_name = os.path.basename(suggested_regions_dir)

            # Create masks and overlays subdirectories in the same directory
            base_dir = os.path.dirname(image_path)
            mask_path = os.path.join(base_dir, 'masks', os.path.basename(
                image_path).replace('.png', '_mask.png'))
            overlay_path = os.path.join(base_dir, 'overlays', os.path.basename(
                image_path).replace('.png', '_overlay.png'))
        else:
            # Extract SVS directory name (parent of 'images' directory)
            # Path structure: .../suggested_regions/SVS_NAME/images/image.png
            path_parts = image_path.split(os.sep)
            images_idx = path_parts.index('images')
            svs_dir_name = path_parts[images_idx - 1]

            # Determine corresponding mask path
            # Replace 'images' with 'masks' and add '_mask' suffix
            mask_path = image_path.replace('/images/', '/masks/')
            mask_path = mask_path.replace('.png', '_mask.png')

            # Determine corresponding overlay path
            # Replace 'images' with 'overlays' and add '_overlay' suffix
            overlay_path = image_path.replace('/images/', '/overlays/')
            overlay_path = overlay_path.replace('.png', '_overlay.png')

        # Check if mask already exists (or if we're reprocessing)
        if reprocess or not os.path.exists(mask_path):
            # Create mask and overlay directories if they don't exist
            mask_dir = os.path.dirname(mask_path)
            overlay_dir = os.path.dirname(overlay_path)
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(overlay_dir, exist_ok=True)

            if verbose:
                print(
                    f"Will process: {os.path.basename(image_path)} (SVS: {svs_dir_name})")
                print(f"  Image: {image_path}")
                print(f"  Mask will be saved to: {mask_path}")
                print(f"  Overlay will be saved to: {overlay_path}")

            # Initialize SVS directory list if not exists
            if svs_dir_name not in images_by_svs:
                images_by_svs[svs_dir_name] = []

            images_by_svs[svs_dir_name].append({
                'image_path': image_path,
                'mask_path': mask_path,
                'overlay_path': overlay_path,
                'image_name': os.path.basename(image_path),
                'svs_dir': svs_dir_name
            })
            total_to_process += 1
        else:
            if not reprocess:
                if verbose:
                    print(
                        f"Mask already exists for {os.path.basename(image_path)} (SVS: {svs_dir_name}), skipping")
                    print(f"  Existing mask: {mask_path}")
                else:
                    print(
                        f"Mask already exists for {os.path.basename(image_path)} (SVS: {svs_dir_name}), skipping")

    print(
        f"Found {total_to_process} images that need prediction across {len(images_by_svs)} SVS directories")
    if verbose and images_by_svs:
        print("Images per SVS directory:")
        for svs_dir, images in images_by_svs.items():
            print(f"  {svs_dir}: {len(images)} images")

    return images_by_svs


def predict_image_from_path(model, image_path, device, transform, verbose=False):
    """
    Perform prediction on a single image from file path.

    Args:
        model: Trained Mask R-CNN model
        image_path (str): Path to the input image
        device: PyTorch device
        transform: Image transformation function
        verbose (bool): Whether to print detailed information

    Returns:
        tuple: (predicted_mask, prediction_time) or (None, None) if failed
    """
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping")
            return None, None

        if verbose:
            print(f"  Loading image from: {image_path}")
            # Check image properties
            img = Image.open(image_path)
            print(f"  Image size: {img.size}, mode: {img.mode}")
            print(f"  Running prediction on device: {device}")

        # Time the prediction
        start_time = time.time()
        result = predict_single_image(
            model, image_path, device, transform, return_time=False)
        prediction_time = time.time() - start_time

        if verbose and result is not None:
            print(f"  Prediction successful, mask shape: {result.shape}")
            print(f"  Mask value range: {result.min()} - {result.max()}")
            unique_vals = np.unique(result)
            print(f"  Unique mask values: {len(unique_vals)} values")
            print(f"  Prediction time: {prediction_time:.3f} seconds")

        return result, prediction_time
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")
        return None, None


def create_overlay_image(image_path, predicted_mask, overlay_path, verbose=False):
    """
    Create an overlay image with the original image and predicted region outline.

    Args:
        image_path (str): Path to original image
        predicted_mask (numpy.ndarray): Predicted mask
        overlay_path (str): Path where overlay will be saved
        verbose (bool): Whether to print detailed information
    """
    try:
        if verbose:
            print(f"  Creating overlay image...")
            print(f"  Loading original image: {image_path}")

        # Load original image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Create binary mask from prediction
        binary_mask = (predicted_mask > 127).astype(np.uint8)

        if verbose:
            print(f"  Binary mask created with threshold 127")
            print(f"  Binary mask has {np.sum(binary_mask)} positive pixels")

        # Find contours of the predicted regions
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if verbose:
            print(f"  Found {len(contours)} contours")

        # Create overlay image
        overlay_img = img_array.copy()

        # Draw contours on the overlay image
        if len(contours) > 0:
            # Draw contours in red with thickness 2
            cv2.drawContours(overlay_img, contours, -1, (255, 255, 0), 3)
            if verbose:
                print(f"  Drew {len(contours)} contours in yellow")
        else:
            if verbose:
                print(f"  No contours to draw")

        # Save overlay image
        overlay_pil = Image.fromarray(overlay_img)
        overlay_pil.save(overlay_path)

        if verbose:
            print(f"  Overlay saved to: {overlay_path}")

    except Exception as e:
        print(f"Error creating overlay for {image_path}: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")


def save_timing_to_tracking_json(image_path, prediction_time, verbose=False):
    """
    Save timing information to a tracking JSON file in the same directory as the image.

    Args:
        image_path (str): Path to the processed image
        prediction_time (float): Time taken for prediction in seconds
        verbose (bool): Whether to print detailed information
    """
    try:
        # Get the directory containing the image
        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)

        # Create tracking JSON filename
        tracking_json_path = os.path.join(image_dir, 'tracking.json')

        # Load existing tracking data or create new
        tracking_data = {}
        if os.path.exists(tracking_json_path):
            try:
                with open(tracking_json_path, 'r') as f:
                    tracking_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                if verbose:
                    print(
                        f"  Warning: Could not load existing tracking.json: {e}")
                    print(f"  Creating new tracking data")
                tracking_data = {}

        # Add timing information for this image
        if image_name not in tracking_data:
            tracking_data[image_name] = {}

        tracking_data[image_name]['prediction_time_seconds'] = round(
            prediction_time, 3)
        tracking_data[image_name]['prediction_timestamp'] = time.strftime(
            '%Y-%m-%d %H:%M:%S')

        # Save updated tracking data
        with open(tracking_json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2, sort_keys=True)

        if verbose:
            print(f"  Timing data saved to: {tracking_json_path}")
            print(f"  Prediction time: {prediction_time:.3f} seconds")

    except Exception as e:
        print(f"Error saving timing data for {image_path}: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")


def save_svs_timing_to_tracking_json(svs_dir_path, svs_processing_time, num_images, verbose=False):
    """
    Save SVS directory processing timing information to a tracking JSON file.

    Args:
        svs_dir_path (str): Path to the SVS directory
        svs_processing_time (float): Time taken to process all images in the SVS directory
        num_images (int): Number of images processed in this SVS directory
        verbose (bool): Whether to print detailed information
    """
    try:
        # Create tracking JSON filename in the SVS directory
        tracking_json_path = os.path.join(svs_dir_path, 'tracking.json')

        # Load existing tracking data or create new
        tracking_data = {}
        if os.path.exists(tracking_json_path):
            try:
                with open(tracking_json_path, 'r') as f:
                    tracking_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                if verbose:
                    print(
                        f"  Warning: Could not load existing tracking.json: {e}")
                    print(f"  Creating new tracking data")
                tracking_data = {}

        # Add SVS directory timing information
        svs_key = '_svs_processing_summary'
        if svs_key not in tracking_data:
            tracking_data[svs_key] = {}

        tracking_data[svs_key]['total_processing_time_seconds'] = round(
            svs_processing_time, 3)
        tracking_data[svs_key]['num_images_processed'] = num_images
        tracking_data[svs_key]['avg_time_per_image_seconds'] = round(
            svs_processing_time / num_images, 3) if num_images > 0 else 0
        tracking_data[svs_key]['processing_timestamp'] = time.strftime(
            '%Y-%m-%d %H:%M:%S')

        # Save updated tracking data
        with open(tracking_json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2, sort_keys=True)

        if verbose:
            print(f"  SVS timing data saved to: {tracking_json_path}")
            print(
                f"  Total processing time: {svs_processing_time:.3f} seconds")
            print(f"  Images processed: {num_images}")
            print(
                f"  Average time per image: {svs_processing_time / num_images:.3f} seconds" if num_images > 0 else "  No images processed")

    except Exception as e:
        print(f"Error saving SVS timing data for {svs_dir_path}: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")


def save_prediction_visualization(image_path, mask_path, predicted_mask, plot_dir):
    """
    Save a visualization of the prediction for quality control.

    Args:
        image_path (str): Path to original image
        mask_path (str): Path where mask will be saved
        predicted_mask (numpy.ndarray): Predicted mask
        plot_dir (str): Directory to save visualization
    """
    try:
        # Create visualization directory
        vis_dir = os.path.join(plot_dir, 'prediction_visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Load original image
        img = Image.open(image_path).convert("RGB")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Predicted mask
        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')

        # Overlay
        overlay = np.array(img)
        mask_binary = predicted_mask > 127
        overlay[mask_binary] = [255, 0, 0]  # Red overlay for vessels

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Red = Vessels)')
        axes[2].axis('off')

        # Save visualization
        vis_filename = os.path.basename(
            image_path).replace('.png', '_prediction.png')
        vis_path = os.path.join(vis_dir, vis_filename)
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Error creating visualization for {image_path}: {e}")


def process_all_images(model_path=None, save_visualizations=False, max_images=None, reprocess=False, verbose=False, custom_dir=None):
    """
    Process all images that need prediction, organized by SVS directory.

    Args:
        model_path (str): Path to saved model
        save_visualizations (bool): Whether to save prediction visualizations
        max_images (int): Maximum number of images to process (for testing)
        reprocess (bool): If True, remove existing outputs and reprocess all images
        verbose (bool): Whether to print detailed information
        custom_dir (str): Custom directory to search for images. If provided, uses this instead of config dir
    """
    if custom_dir:
        print(f"Starting batch prediction on custom directory: {custom_dir}")
    else:
        print("Starting batch prediction on suggested regions...")

    if verbose:
        print(f"Configuration:")
        print(f"  Data directory: {data_dir}")
        print(f"  Artifacts directory: {artifacts_dir}")
        print(f"  Plot directory: {plot_dir}")
        print(
            f"  Model path: {model_path if model_path else 'default best_val_mask_rcnn_model.pth'}")
        print(f"  Save visualizations: {save_visualizations}")
        print(f"  Max images: {max_images if max_images else 'unlimited'}")
        print(f"  Reprocess existing: {reprocess}")
        print(
            f"  Custom directory: {custom_dir if custom_dir else 'None (using config)'}")

    # Load model
    if verbose:
        print("Loading model...")
    model, device, transform = load_model(model_path)
    print(
        f"Model loaded successfully from {model_path if model_path else 'default best_val_mask_rcnn_model.pth'}")
    print(f"Using device: {device}")

    # Find images to process (grouped by SVS directory)
    if verbose:
        print("Scanning for images to process...")
    images_by_svs = find_images_to_process(
        reprocess=reprocess, verbose=verbose, custom_dir=custom_dir)

    if not images_by_svs:
        print("No images found that need prediction.")
        return

    # Calculate total images and apply max_images limit if specified
    total_images = sum(len(images) for images in images_by_svs.values())
    if max_images and max_images < total_images:
        print(f"Limited processing to {max_images} images for testing")
        # Apply limit by truncating image lists
        remaining_limit = max_images
        for svs_dir in list(images_by_svs.keys()):
            if remaining_limit <= 0:
                del images_by_svs[svs_dir]
            elif len(images_by_svs[svs_dir]) > remaining_limit:
                images_by_svs[svs_dir] = images_by_svs[svs_dir][:remaining_limit]
                remaining_limit = 0
            else:
                remaining_limit -= len(images_by_svs[svs_dir])

    # Process images grouped by SVS directory
    successful_predictions = 0
    failed_predictions = 0
    total_processed = 0

    # Double loop: first over SVS directories, then over images within each directory
    for svs_dir_name, images_list in images_by_svs.items():
        print(
            f"\nProcessing SVS directory: {svs_dir_name} ({len(images_list)} images)")

        # Remove existing outputs if reprocessing
        if reprocess and images_list:
            # Get SVS directory path from first image
            first_image_path = images_list[0]['image_path']
            # Go up from images/ to SVS directory
            svs_dir_path = os.path.dirname(os.path.dirname(first_image_path))

            if verbose:
                print(f"  Reprocessing enabled - removing existing outputs...")
            remove_svs_outputs(svs_dir_path, verbose=verbose)

        if verbose:
            print(f"  Directory contains {len(images_list)} images to process")

        # Start timing for this SVS directory
        svs_start_time = time.time()
        svs_successful_count = 0

        # Process all images in this SVS directory
        for i, item in enumerate(tqdm(images_list, desc=f"Processing {svs_dir_name}", leave=False)):
            image_path = item['image_path']
            mask_path = item['mask_path']
            overlay_path = item['overlay_path']
            image_name = item['image_name']
            total_processed += 1

            if verbose:
                print(
                    f"\n  Processing image {i+1}/{len(images_list)} in {svs_dir_name}: {image_name}")

            # Perform prediction
            predicted_mask, prediction_time = predict_image_from_path(
                model, image_path, device, transform, verbose=verbose)

            if predicted_mask is not None:
                # Save timing information to tracking JSON
                save_timing_to_tracking_json(
                    image_path, prediction_time, verbose=verbose)
                try:
                    if verbose:
                        print(f"    Saving predicted mask...")

                    # Save predicted mask
                    mask_img = Image.fromarray(predicted_mask, mode='L')
                    mask_img.save(mask_path)

                    if verbose:
                        print(f"    Mask saved to: {mask_path}")

                    # Create and save overlay image
                    create_overlay_image(
                        image_path, predicted_mask, overlay_path, verbose=verbose)

                    # Save visualization if requested
                    if save_visualizations:
                        if verbose:
                            print(f"    Creating visualization...")
                        save_prediction_visualization(
                            image_path, mask_path, predicted_mask, plot_dir)

                    successful_predictions += 1
                    svs_successful_count += 1
                    if verbose:
                        print(
                            f"    ✓ Successfully processed {image_name} in {prediction_time:.3f}s")
                    else:
                        print(
                            f"  Saved mask and overlay for {image_name} ({prediction_time:.3f}s)")

                except Exception as e:
                    print(f"  Error saving outputs for {image_name}: {e}")
                    if verbose:
                        import traceback
                        print(f"    Full traceback: {traceback.format_exc()}")
                    failed_predictions += 1
            else:
                if verbose:
                    print(f"    ✗ Prediction failed for {image_name}")
                failed_predictions += 1

        # Calculate SVS directory processing time
        svs_processing_time = time.time() - svs_start_time

        # Get SVS directory path for saving timing data
        if images_list:
            # Get the parent directory of the images directory
            first_image_path = images_list[0]['image_path']
            # Go up from images/ to SVS directory
            svs_dir_path = os.path.dirname(os.path.dirname(first_image_path))

            # Save SVS timing information
            save_svs_timing_to_tracking_json(
                svs_dir_path, svs_processing_time, svs_successful_count, verbose=verbose)

        print(
            f"Completed SVS directory: {svs_dir_name} ({svs_processing_time:.1f}s, {svs_successful_count} successful)")

    print(f"\nBatch prediction complete!")
    print(f"Processed {len(images_by_svs)} SVS directories")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Total processed: {total_processed}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch prediction on suggested regions or custom directory')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model (default: best_val_mask_rcnn_model.pth)')
    parser.add_argument('--dir', type=str, default=None,
                        help='Arbitrary directory containing images to process. If provided, uses this instead of config dir')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for predictions when using --dir (default: <dir>/predictions)')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='Save prediction visualizations for quality control')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information during processing')
    parser.add_argument('--reprocess', action='store_true',
                        help='Remove existing mask and overlay directories and reprocess all images')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    process_all_images(
        model_path=args.model_path,
        save_visualizations=args.save_visualizations,
        max_images=args.max_images,
        reprocess=args.reprocess,
        verbose=args.verbose,
        custom_dir=args.dir
    )


if __name__ == "__main__":
    main()
