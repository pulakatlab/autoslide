"""
Visualization script for data augmentation transformations.

This script creates a grid plot showing examples of each augmentation transformation
applied to sample images and their corresponding masks. This helps verify that
transformations are working correctly and understand their effects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from autoslide import config
from autoslide.pipeline.model.data_preprocessing import (
    load_data, 
    RandomShear, 
    RandomRotation90,
    generate_negative_samples,
    generate_artificial_vessels,
    get_mask_outline
)
from torchvision.transforms import v2 as T


def apply_individual_transforms(img, mask):
    """
    Apply individual transformations to an image and mask pair.
    
    Args:
        img (PIL.Image): Input image
        mask (PIL.Image): Input mask
        
    Returns:
        dict: Dictionary containing transformed image-mask pairs for each transformation
    """
    transforms_dict = {}
    
    # Original (no transformation)
    transforms_dict['Original'] = (img, mask)
    
    # Horizontal flip
    h_flip = T.RandomHorizontalFlip(p=1.0)
    img_h, mask_h = h_flip(img, mask)
    transforms_dict['Horizontal Flip'] = (img_h, mask_h)
    
    # Vertical flip
    v_flip = T.RandomVerticalFlip(p=1.0)
    img_v, mask_v = v_flip(img, mask)
    transforms_dict['Vertical Flip'] = (img_v, mask_v)
    
    # 90 degree rotation
    rot_90 = RandomRotation90(p=1.0)
    img_r, mask_r = rot_90(img, mask)
    transforms_dict['90° Rotation'] = (img_r, mask_r)
    
    # Shear transformation
    shear = RandomShear(p=1.0, shear_range=20)
    img_s, mask_s = shear(img, mask)
    transforms_dict['Shear'] = (img_s, mask_s)
    
    # Color jitter (only affects image)
    color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    img_c = color_jitter(img)
    transforms_dict['Color Jitter'] = (img_c, mask)
    
    return transforms_dict


def apply_augmentation_transforms(img_np, mask_np):
    """
    Apply augmentation transformations that generate new samples.
    
    Args:
        img_np (numpy.ndarray): Input image as numpy array
        mask_np (numpy.ndarray): Input mask as numpy array
        
    Returns:
        dict: Dictionary containing augmented image-mask pairs
    """
    augment_dict = {}
    
    # Original
    augment_dict['Original'] = (img_np, mask_np)
    
    # Negative samples
    neg_img, neg_mask = generate_negative_samples(img_np, mask_np)
    augment_dict['Negative Sample'] = (neg_img, neg_mask)
    
    # Artificial vessels
    art_img, art_mask = generate_artificial_vessels(img_np, mask_np)
    if art_img is not None and art_mask is not None:
        augment_dict['Artificial Vessels'] = (art_img, art_mask)
    else:
        # If no vessels to augment, show original
        augment_dict['Artificial Vessels'] = (img_np, mask_np)
    
    return augment_dict


def create_transformation_grid(img_dir, mask_dir, image_names, mask_names, 
                             output_path, sample_idx=0):
    """
    Create a grid plot showing all transformation examples.
    
    Args:
        img_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        image_names (list): List of image filenames
        mask_names (list): List of mask filenames
        output_path (str): Path to save the output plot
        sample_idx (int): Index of the sample to use for visualization
    """
    # Load sample image and mask
    img = Image.open(os.path.join(img_dir, image_names[sample_idx])).convert("RGB")
    mask = Image.open(os.path.join(mask_dir, mask_names[sample_idx])).convert("L")
    
    # Convert to numpy for augmentation functions
    img_np = np.array(img)
    mask_np = np.array(mask)
    
    # Apply individual transforms
    individual_transforms = apply_individual_transforms(img, mask)
    
    # Apply augmentation transforms
    augmentation_transforms = apply_augmentation_transforms(img_np, mask_np)
    
    # Combine all transforms
    all_transforms = {**individual_transforms, **augmentation_transforms}
    
    # Remove duplicates (Original appears in both)
    unique_transforms = {}
    seen_keys = set()
    for key, value in all_transforms.items():
        if key not in seen_keys:
            unique_transforms[key] = value
            seen_keys.add(key)
    
    # Create grid plot
    n_transforms = len(unique_transforms)
    n_cols = 3  # 3 columns: original image, transformed image, mask overlay
    n_rows = n_transforms
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Data Augmentation Transformations\nSample: {image_names[sample_idx]}', 
                 fontsize=16, y=0.98)
    
    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (transform_name, (trans_img, trans_mask)) in enumerate(unique_transforms.items()):
        # Convert PIL images to numpy if needed
        if isinstance(trans_img, Image.Image):
            trans_img = np.array(trans_img)
        if isinstance(trans_mask, Image.Image):
            trans_mask = np.array(trans_mask)
        
        # Original/transformed image
        axes[i, 0].imshow(trans_img)
        axes[i, 0].set_title(f'{transform_name}\nImage')
        axes[i, 0].axis('off')
        
        # Mask
        axes[i, 1].imshow(trans_mask, cmap='gray')
        axes[i, 1].set_title(f'{transform_name}\nMask')
        axes[i, 1].axis('off')
        
        # Overlay (image with mask outline)
        axes[i, 2].imshow(trans_img)
        if np.max(trans_mask) > 0:
            mask_outline = get_mask_outline(trans_mask > 0)
            y_coords, x_coords = np.where(mask_outline > 0)
            axes[i, 2].scatter(x_coords, y_coords, c='yellow', s=0.5, alpha=0.8)
        axes[i, 2].set_title(f'{transform_name}\nOverlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Transformation grid saved to: {output_path}")


def create_multiple_samples_grid(img_dir, mask_dir, image_names, mask_names, 
                               output_path, n_samples=3):
    """
    Create a grid showing the same transformation applied to multiple samples.
    
    Args:
        img_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        image_names (list): List of image filenames
        mask_names (list): List of mask filenames
        output_path (str): Path to save the output plot
        n_samples (int): Number of different samples to show
    """
    # Select random samples
    sample_indices = np.random.choice(len(image_names), n_samples, replace=False)
    
    # Define a few key transformations to show
    key_transforms = ['Original', 'Horizontal Flip', 'Shear', 'Negative Sample', 'Artificial Vessels']
    
    fig, axes = plt.subplots(len(key_transforms), n_samples * 2, 
                           figsize=(4 * n_samples * 2, 4 * len(key_transforms)))
    fig.suptitle('Augmentation Transformations Across Multiple Samples', 
                 fontsize=16, y=0.98)
    
    # Ensure axes is 2D
    if len(key_transforms) == 1:
        axes = axes.reshape(1, -1)
    if n_samples * 2 == 1:
        axes = axes.reshape(-1, 1)
    
    for i, transform_name in enumerate(key_transforms):
        for j, sample_idx in enumerate(sample_indices):
            # Load sample
            img = Image.open(os.path.join(img_dir, image_names[sample_idx])).convert("RGB")
            mask = Image.open(os.path.join(mask_dir, mask_names[sample_idx])).convert("L")
            img_np = np.array(img)
            mask_np = np.array(mask)
            
            # Apply transformation
            if transform_name == 'Original':
                trans_img, trans_mask = img_np, mask_np
            elif transform_name == 'Horizontal Flip':
                h_flip = T.RandomHorizontalFlip(p=1.0)
                trans_img_pil, trans_mask_pil = h_flip(img, mask)
                trans_img, trans_mask = np.array(trans_img_pil), np.array(trans_mask_pil)
            elif transform_name == 'Shear':
                shear = RandomShear(p=1.0, shear_range=20)
                trans_img_pil, trans_mask_pil = shear(img, mask)
                trans_img, trans_mask = np.array(trans_img_pil), np.array(trans_mask_pil)
            elif transform_name == 'Negative Sample':
                trans_img, trans_mask = generate_negative_samples(img_np, mask_np)
            elif transform_name == 'Artificial Vessels':
                trans_img, trans_mask = generate_artificial_vessels(img_np, mask_np)
                if trans_img is None:
                    trans_img, trans_mask = img_np, mask_np
            
            # Plot image
            col_img = j * 2
            axes[i, col_img].imshow(trans_img)
            if i == 0:  # Only add title to top row
                axes[i, col_img].set_title(f'Sample {sample_idx}\nImage')
            axes[i, col_img].axis('off')
            
            # Plot mask with overlay
            col_mask = j * 2 + 1
            axes[i, col_mask].imshow(trans_img)
            if np.max(trans_mask) > 0:
                mask_outline = get_mask_outline(trans_mask > 0)
                y_coords, x_coords = np.where(mask_outline > 0)
                axes[i, col_mask].scatter(x_coords, y_coords, c='yellow', s=0.5, alpha=0.8)
            if i == 0:  # Only add title to top row
                axes[i, col_mask].set_title(f'Sample {sample_idx}\nOverlay')
            axes[i, col_mask].axis('off')
        
        # Add row label
        axes[i, 0].text(-0.1, 0.5, transform_name, transform=axes[i, 0].transAxes,
                       rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Multiple samples grid saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize data augmentation transformations')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (uses config default if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (uses config plot_dirs if not specified)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to use for single sample visualization')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='Number of samples to show in multi-sample visualization')
    return parser.parse_args()


def main():
    """Main function to generate augmentation visualization plots."""
    args = parse_args()
    
    # Set up directories
    data_dir = args.data_dir if args.data_dir else config['data_dir']
    output_dir = args.output_dir if args.output_dir else config['plot_dirs']
    
    print("Loading data for augmentation visualization...")
    
    # Load data
    labelled_data_dir, img_dir, mask_dir, image_names, mask_names = load_data(data_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating augmentation visualizations...")
    print(f"Using sample index: {args.sample_idx}")
    print(f"Output directory: {output_dir}")
    
    # Create single sample transformation grid
    single_sample_path = os.path.join(output_dir, 'augmentation_transformations_single.png')
    create_transformation_grid(img_dir, mask_dir, image_names, mask_names, 
                             single_sample_path, args.sample_idx)
    
    # Create multiple samples grid
    multi_sample_path = os.path.join(output_dir, 'augmentation_transformations_multi.png')
    create_multiple_samples_grid(img_dir, mask_dir, image_names, mask_names, 
                               multi_sample_path, args.n_samples)
    
    print("\nAugmentation visualization complete!")
    print(f"Generated plots:")
    print(f"  - Single sample transformations: {single_sample_path}")
    print(f"  - Multiple sample transformations: {multi_sample_path}")


if __name__ == "__main__":
    main()
