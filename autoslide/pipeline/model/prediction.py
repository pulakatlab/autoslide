"""
Batch prediction script for vessel segmentation on all images in suggested_regions.

This script:
1. Finds all images under data_dir/suggested_regions/*/images/*.png
2. Checks if corresponding masks already exist
3. Performs prediction only on images without existing masks
4. Saves predicted masks to corresponding mask directories
"""

from autoslide.pipeline import utils
import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import cv2

from autoslide import config
from autoslide.pipeline.model.prediction_utils import load_model, predict_single_image

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
plot_dir = config['plot_dirs']


##############################

def find_images_to_process(verbose=False):
    """
    Find all images that need prediction and don't already have masks.

    Args:
        verbose (bool): Whether to print detailed information

    Returns:
        list: List of dictionaries containing image paths and corresponding mask and overlay paths
    """
    suggested_regions_dir = os.path.join(data_dir, 'suggested_regions')

    if verbose:
        print(f"Looking for suggested regions in: {suggested_regions_dir}")

    if not os.path.exists(suggested_regions_dir):
        print(
            f"Suggested regions directory not found: {suggested_regions_dir}")
        return []

    # Find all image files
    image_pattern = os.path.join(
        suggested_regions_dir, '**', 'images', '*.png')
    image_paths = glob(image_pattern, recursive=True)

    if verbose:
        print(f"Search pattern: {image_pattern}")
        print(f"Found {len(image_paths)} images in suggested_regions")
        if image_paths:
            print("Sample image paths:")
            for i, path in enumerate(image_paths[:3]):
                print(f"  {i+1}. {path}")
            if len(image_paths) > 3:
                print(f"  ... and {len(image_paths) - 3} more")
    else:
        print(f"Found {len(image_paths)} images in suggested_regions")

    images_to_process = []

    for image_path in image_paths:
        # Determine corresponding mask path
        # Replace 'images' with 'masks' and add '_mask' suffix
        mask_path = image_path.replace('/images/', '/masks/')
        mask_path = mask_path.replace('.png', '_mask.png')

        # Determine corresponding overlay path
        # Replace 'images' with 'overlays' and add '_overlay' suffix
        overlay_path = image_path.replace('/images/', '/overlays/')
        overlay_path = overlay_path.replace('.png', '_overlay.png')

        # Check if mask already exists
        if not os.path.exists(mask_path):
            # Create mask and overlay directories if they don't exist
            mask_dir = os.path.dirname(mask_path)
            overlay_dir = os.path.dirname(overlay_path)
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(overlay_dir, exist_ok=True)

            if verbose:
                print(f"Will process: {os.path.basename(image_path)}")
                print(f"  Image: {image_path}")
                print(f"  Mask will be saved to: {mask_path}")
                print(f"  Overlay will be saved to: {overlay_path}")

            images_to_process.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'overlay_path': overlay_path,
                'image_name': os.path.basename(image_path)
            })
        else:
            if verbose:
                print(f"Mask already exists for {os.path.basename(image_path)}, skipping")
                print(f"  Existing mask: {mask_path}")
            else:
                print(
                    f"Mask already exists for {os.path.basename(image_path)}, skipping")

    print(f"Found {len(images_to_process)} images that need prediction")
    return images_to_process


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
        numpy.ndarray: Combined predicted mask
    """
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping")
            return None
            
        if verbose:
            print(f"  Loading image from: {image_path}")
            # Check image properties
            img = Image.open(image_path)
            print(f"  Image size: {img.size}, mode: {img.mode}")
            print(f"  Running prediction on device: {device}")
        
        result = predict_single_image(model, image_path, device, transform, return_time=False)
        
        if verbose and result is not None:
            print(f"  Prediction successful, mask shape: {result.shape}")
            print(f"  Mask value range: {result.min()} - {result.max()}")
            unique_vals = np.unique(result)
            print(f"  Unique mask values: {len(unique_vals)} values")
        
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        if verbose:
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")
        return None


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


def process_all_images(model_path=None, save_visualizations=False, max_images=None, verbose=False):
    """
    Process all images that need prediction.

    Args:
        model_path (str): Path to saved model
        save_visualizations (bool): Whether to save prediction visualizations
        max_images (int): Maximum number of images to process (for testing)
        verbose (bool): Whether to print detailed information
    """
    print("Starting batch prediction on suggested regions...")
    
    if verbose:
        print(f"Configuration:")
        print(f"  Data directory: {data_dir}")
        print(f"  Artifacts directory: {artifacts_dir}")
        print(f"  Plot directory: {plot_dir}")
        print(f"  Model path: {model_path if model_path else 'default best_val_mask_rcnn_model.pth'}")
        print(f"  Save visualizations: {save_visualizations}")
        print(f"  Max images: {max_images if max_images else 'unlimited'}")

    # Load model
    if verbose:
        print("Loading model...")
    model, device, transform = load_model(model_path)
    print(
        f"Model loaded successfully from {model_path if model_path else 'default best_val_mask_rcnn_model.pth'}")
    print(f"Using device: {device}")

    # Find images to process
    if verbose:
        print("Scanning for images to process...")
    images_to_process = find_images_to_process(verbose=verbose)

    if not images_to_process:
        print("No images found that need prediction.")
        return

    # Limit number of images if specified
    if max_images and max_images < len(images_to_process):
        images_to_process = images_to_process[:max_images]
        print(f"Limited processing to {max_images} images for testing")

    # Process each image
    successful_predictions = 0
    failed_predictions = 0

    for i, item in enumerate(tqdm(images_to_process, desc="Processing images")):
        image_path = item['image_path']
        mask_path = item['mask_path']
        overlay_path = item['overlay_path']
        image_name = item['image_name']

        if verbose:
            print(f"\nProcessing image {i+1}/{len(images_to_process)}: {image_name}")

        # Perform prediction
        predicted_mask = predict_image_from_path(
            model, image_path, device, transform, verbose=verbose)

        if predicted_mask is not None:
            try:
                if verbose:
                    print(f"  Saving predicted mask...")
                
                # Save predicted mask
                mask_img = Image.fromarray(predicted_mask, mode='L')
                mask_img.save(mask_path)
                
                if verbose:
                    print(f"  Mask saved to: {mask_path}")

                # Create and save overlay image
                create_overlay_image(image_path, predicted_mask, overlay_path, verbose=verbose)

                # Save visualization if requested
                if save_visualizations:
                    if verbose:
                        print(f"  Creating visualization...")
                    save_prediction_visualization(
                        image_path, mask_path, predicted_mask, plot_dir)

                successful_predictions += 1
                if verbose:
                    print(f"  ✓ Successfully processed {image_name}")
                else:
                    print(
                        f"Saved mask and overlay for {image_name}, parent_dir: {os.path.dirname(mask_path)}")

            except Exception as e:
                print(f"Error saving outputs for {image_name}: {e}")
                if verbose:
                    import traceback
                    print(f"  Full traceback: {traceback.format_exc()}")
                failed_predictions += 1
        else:
            if verbose:
                print(f"  ✗ Prediction failed for {image_name}")
            failed_predictions += 1

    print(f"\nBatch prediction complete!")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Total processed: {successful_predictions + failed_predictions}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch prediction on suggested regions')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model (default: best_val_mask_rcnn_model.pth)')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='Save prediction visualizations for quality control')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information during processing')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    process_all_images(
        model_path=args.model_path,
        save_visualizations=args.save_visualizations,
        max_images=args.max_images,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
