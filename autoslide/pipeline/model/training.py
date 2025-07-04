"""
Resources
- https://github.com/DatumLearning/Mask-RCNN-finetuning-PyTorch/blob/main/notebook.ipynb
- https://www.youtube.com/watch?v=vV9L71hK-RE
- https://www.youtube.com/watch?v=t1MrzuAUdoE

"""

#!/usr/bin/env python
# coding: utf-8


from autoslide.pipeline.model.training_utils import (
    setup_directories, load_data, get_mask_outline, RandomRotation90,
    create_transforms, test_transformations, CustDat, initialize_model,
    custom_collate, split_train_val, load_or_create_augmented_data,
    load_negative_images, plot_augmented_samples, combine_datasets,
    create_sample_plots, AugmentedCustDat, create_dataloaders,
    setup_training, train_model, plot_losses, evaluate_model, load_model,
    generate_negative_samples, generate_artificial_vessels, augment_dataset
)
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import cv2

import torch
import torchvision
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import config
from autoslide import config

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = config['artifacts_dir']
plot_dir = config['plot_dirs']

# Import utilities directly

##############################
##############################
retrain_bool = False

##############################
##############################


def main():
    """Main function to run the training pipeline"""
    # Setup directories
    plot_dir, artifacts_dir = setup_directories(data_dir)

    # Load data
    labelled_data_dir, img_dir, mask_dir, image_names, mask_names = load_data(
        data_dir)

    # Create transforms
    transform = create_transforms()

    # Optional: Test transformations
    # test_transformations(img_dir, mask_dir, image_names, mask_names, transform)

    # Split data into train and validation sets
    train_imgs, train_masks, val_imgs, val_masks = split_train_val(
        image_names, mask_names)

    # Load or create augmented data
    aug_img_dir, aug_mask_dir, aug_img_names, aug_mask_names = load_or_create_augmented_data(
        labelled_data_dir, img_dir, mask_dir, train_imgs, train_masks
    )

    # Load negative images
    neg_image_dir, neg_mask_dir, neg_img_names, neg_mask_names = load_negative_images(
        labelled_data_dir)

    # Plot augmented samples
    plot_augmented_samples(aug_img_dir, aug_mask_dir,
                           aug_img_names, aug_mask_names, plot_dir)

    # Combine datasets
    train_imgs, train_masks, val_imgs, val_masks = combine_datasets(
        train_imgs, train_masks, val_imgs, val_masks,
        aug_img_names, aug_mask_names, neg_img_names, neg_mask_names
    )

    # Create sample plots
    create_sample_plots(
        train_imgs, train_masks, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        neg_image_dir, neg_mask_dir, plot_dir
    )

    # Create dataloaders
    train_dl, val_dl = create_dataloaders(
        train_imgs, train_masks, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        neg_image_dir, neg_mask_dir, transform
    )

    # Initialize model
    model = initialize_model()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Setup training
    optimizer = setup_training(model, device)

    # Model paths
    best_model_path = os.path.join(
        artifacts_dir, 'best_val_mask_rcnn_model.pth')

    # Load existing model or train new one
    if os.path.exists(best_model_path) and not retrain_bool:
        print('Loading model from savefile')
        model = load_model(model, best_model_path, device)
    else:
        # Train model
        model, all_train_losses, all_val_losses, best_val_loss = train_model(
            model, train_dl, val_dl, optimizer, device, plot_dir, artifacts_dir
        )

    # Evaluate model
    evaluate_model(
        model, val_imgs, val_masks, neg_img_names, neg_mask_names,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        neg_image_dir, neg_mask_dir, device, plot_dir
    )


if __name__ == "__main__":
    main()
