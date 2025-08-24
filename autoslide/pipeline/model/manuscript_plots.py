import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2 as cv
from torchvision.transforms import v2 as T
from tqdm import tqdm
import argparse

from autoslide.pipeline.model.prediction_utils import load_model, setup_device
from autoslide.pipeline.model.data_preprocessing import load_data
from autoslide import config


def create_manuscript_plots(model, val_imgs, val_masks, img_dir, mask_dir, 
                           aug_img_dir, aug_mask_dir, device, output_dir, 
                           max_samples=None):
    """
    Create manuscript-quality prediction plots.
    
    For each image, creates 2 subplots:
    1. Raw image with yellow outline around predicted vessels
    2. Raw image with green overlay on entire area except predicted vessels
    
    Args:
        model: Trained Mask R-CNN model
        val_imgs: List of validation image filenames
        val_masks: List of validation mask filenames  
        img_dir: Directory containing original images
        mask_dir: Directory containing original masks
        aug_img_dir: Directory containing augmented images
        aug_mask_dir: Directory containing augmented masks
        device: Device to run model on
        output_dir: Directory to save plots
        max_samples: Maximum number of samples to process (None for all)
    """
    print('Creating manuscript plots...')
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    transform = T.ToTensor()
    
    # Limit samples if specified
    if max_samples:
        val_imgs = val_imgs[:max_samples]
        val_masks = val_masks[:max_samples]
    
    print(f'Processing {len(val_imgs)} images...')
    
    for img_name, mask_name in tqdm(zip(val_imgs, val_masks), total=len(val_imgs)):
        # Load image
        if 'aug_' in img_name:
            img = Image.open(os.path.join(aug_img_dir, img_name)).convert("RGB")
        else:
            img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        
        img_array = np.array(img)
        
        # Get prediction
        img_tensor = transform(img)
        with torch.no_grad():
            pred = model([img_tensor.to(device)])
        
        # Combine all predicted masks
        combined_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        if len(pred[0]["masks"]) > 0:
            for mask_tensor in pred[0]["masks"]:
                mask_np = (mask_tensor.cpu().detach().numpy() > 0.5).squeeze().astype(np.uint8)
                combined_mask = np.logical_or(combined_mask, mask_np).astype(np.uint8)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Subplot 1: Raw image with yellow outline
        axes[0].imshow(img_array)
        if combined_mask.sum() > 0:
            # Find contours for outline
            contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Convert contour to matplotlib format
                contour_points = contour.squeeze()
                if len(contour_points.shape) == 2 and contour_points.shape[0] > 2:
                    # Close the contour by adding first point at the end
                    contour_closed = np.vstack([contour_points, contour_points[0]])
                    axes[0].plot(contour_closed[:, 0], contour_closed[:, 1], 
                               color='yellow', linewidth=2)
        
        axes[0].set_title('Predicted Vessel Outline')
        axes[0].axis('off')
        
        # Subplot 2: Raw image with green overlay except on predicted areas
        axes[1].imshow(img_array)
        if combined_mask.sum() > 0:
            # Create green overlay mask (inverse of prediction)
            overlay_mask = 1 - combined_mask
            # Create green overlay
            green_overlay = np.zeros_like(img_array)
            green_overlay[:, :, 1] = 255  # Green channel
            
            # Apply overlay with transparency
            alpha = 0.3
            for c in range(3):
                img_array_overlay = img_array.copy()
                img_array_overlay[:, :, c] = (
                    (1 - alpha * overlay_mask) * img_array[:, :, c] + 
                    alpha * overlay_mask * green_overlay[:, :, c]
                )
            axes[1].imshow(img_array_overlay.astype(np.uint8))
        else:
            # If no prediction, show full green overlay
            green_overlay = img_array.copy()
            green_overlay[:, :, 1] = np.clip(green_overlay[:, :, 1] + 80, 0, 255)
            axes[1].imshow(green_overlay)
        
        axes[1].set_title('Non-vessel Areas (Green Overlay)')
        axes[1].axis('off')
        
        # Save plot
        base_name = os.path.splitext(img_name)[0]
        output_path = os.path.join(output_dir, f'{base_name}_manuscript.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f'Manuscript plots saved to: {output_dir}')


def main():
    """Main function to run manuscript plot generation."""
    parser = argparse.ArgumentParser(description='Generate manuscript plots for vessel predictions')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (default: best model from artifacts)')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to process (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: plot_dir/manuscript_plots)')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    data_dir = config['data_dir']
    artifacts_dir = config['artifacts_dir']
    plot_dir = config['plot_dirs']
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(plot_dir, 'manuscript_plots')
    
    # Load data
    print('Loading data...')
    labelled_data_dir = os.path.join(data_dir, 'labelled_images')
    img_dir = os.path.join(labelled_data_dir, 'images') + '/'
    mask_dir = os.path.join(labelled_data_dir, 'masks') + '/'
    aug_img_dir = os.path.join(labelled_data_dir, 'aug_images') + '/'
    aug_mask_dir = os.path.join(labelled_data_dir, 'aug_masks') + '/'
    
    # Load image and mask lists
    train_imgs, train_masks, val_imgs, val_masks = load_data(data_dir)
    
    # Load model
    print('Loading model...')
    model = load_model(args.model_path, device)
    
    # Create plots
    create_manuscript_plots(
        model=model,
        val_imgs=val_imgs,
        val_masks=val_masks,
        img_dir=img_dir,
        mask_dir=mask_dir,
        aug_img_dir=aug_img_dir,
        aug_mask_dir=aug_mask_dir,
        device=device,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    print('Manuscript plot generation complete!')


if __name__ == '__main__':
    main()
