"""
Data preprocessing and augmentation pipeline for vessel segmentation training.

This module handles:
- Loading original images and masks
- Data augmentation (negative samples, artificial vessels, transforms)
- Train/validation splitting
- Dataset creation and DataLoader setup
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2 as cv
from torchvision.transforms import v2 as T
from tqdm import tqdm, trange
from autoslide import config


#############################################################################
# Data Loading Functions
#############################################################################

def load_data(data_dir=None):
    """
    Load original image and mask data from the labelled_images directory.

    Args:
        data_dir (str): Root data directory

    Returns:
        tuple: (labelled_data_dir, img_dir, mask_dir, image_names, mask_names) -
               Directories and lists of image and mask filenames
    """
    if data_dir is None:
        data_dir = config['data_dir']

    labelled_data_dir = os.path.join(data_dir, 'labelled_images')
    img_dir = os.path.join(labelled_data_dir, 'images/')
    mask_dir = os.path.join(labelled_data_dir, 'masks/')
    image_names = sorted(os.listdir(img_dir))
    mask_names = sorted(os.listdir(mask_dir))

    # Validate that masks correspond to images
    for img_path, mask_path in zip(image_names, mask_names):
        assert img_path.split(".")[0] in mask_path

    print(f'Found {len(image_names)} images and {len(mask_names)} masks')
    print(f'Image directory: {img_dir}')
    print(f'Mask directory: {mask_dir}')

    return labelled_data_dir, img_dir, mask_dir, image_names, mask_names


def split_train_val(image_names, mask_names, train_ratio=0.9):
    """
    Split the dataset into training and validation sets.

    Randomly selects 90% of the data for training and 10% for validation,
    ensuring that the training set has an even number of samples.

    Args:
        image_names (list): List of image filenames
        mask_names (list): List of mask filenames
        train_ratio (float): Ratio of data to use for training

    Returns:
        tuple: (train_imgs, train_masks, val_imgs, val_masks) -
               Lists of filenames for training and validation
    """
    print(
        f'Splitting dataset: {len(image_names)} total images with {train_ratio:.1%} for training')
    num = int(train_ratio * len(image_names))
    num = num if num % 2 == 0 else num + 1
    print(
        f'Adjusted training set size to {num} (even number for batch processing)')

    train_imgs_inds = np.random.choice(
        range(len(image_names)), num, replace=False)
    val_imgs_inds = np.setdiff1d(range(len(image_names)), train_imgs_inds)
    train_imgs = np.array(image_names)[train_imgs_inds]
    val_imgs = np.array(image_names)[val_imgs_inds]
    train_masks = np.array(mask_names)[train_imgs_inds]
    val_masks = np.array(mask_names)[val_imgs_inds]

    print(f'Training images: {len(train_imgs)}')
    print(f'Validation images: {len(val_imgs)}')

    return train_imgs, train_masks, val_imgs, val_masks


#############################################################################
# Data Augmentation Functions
#############################################################################

def get_mask_outline(mask):
    """
    Extract the outline of a mask for visualization purposes.

    Args:
        mask (numpy.ndarray): Binary mask image

    Returns:
        numpy.ndarray: Outline of the mask as a binary image
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask_outline = np.zeros_like(mask)
    mask_outline = cv.drawContours(mask_outline, contours, -1, 255, 1)
    return mask_outline


class RandomShear():
    """
    Randomly shear images with a given probability.

    This transformation is applied to both the image and its corresponding mask.
    """

    def __init__(self, p=0.5, shear_range=20):
        """
        Initialize the random shear transform.

        Args:
            p (float): Probability of applying the shear
            shear_range (int): Maximum shear angle in degrees
        """
        self.p = p
        self.shear_range = shear_range

    def __call__(self, img, mask):
        """
        Apply random shear to image and mask.

        Args:
            img (PIL.Image): Input image
            mask (PIL.Image): Corresponding mask

        Returns:
            tuple: (sheared_img, sheared_mask)
        """
        if np.random.rand() < self.p:
            shear_angle = np.random.uniform(-self.shear_range,
                                            self.shear_range)
            img = img.transform(img.size, Image.AFFINE, (1, np.tan(
                np.radians(shear_angle)), 0, 0, 1, 0))
            mask = mask.transform(
                mask.size, Image.AFFINE, (1, np.tan(np.radians(shear_angle)), 0, 0, 1, 0))
        return img, mask


class RandomRotation90():
    """
    Randomly rotate images by 90 or 270 degrees with probability p.

    This transformation is applied to both the image and its corresponding mask.
    """

    def __init__(self, p=0.5):
        """
        Initialize the random rotation transform.

        Args:
            p (float): Probability of applying the rotation
        """
        self.p = p

    def __call__(self, img, mask):
        """
        Apply random rotation to image and mask.

        Args:
            img (PIL.Image): Input image
            mask (PIL.Image): Corresponding mask

        Returns:
            tuple: (rotated_img, rotated_mask)
        """
        if np.random.rand() < self.p:
            angle = np.random.choice([90, 270])
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask


def create_transforms():
    """
    Create image transformations for data augmentation.

    Creates a composition of transforms including horizontal/vertical flips,
    90-degree rotations, and color jitter, followed by conversion to tensor.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline
    """
    transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        RandomRotation90(p=0.5),
        RandomShear(p=0.5, shear_range=20),  # Added shear transformation
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ElasticTransform(alpha=3000, sigma=30),
        T.ToTensor()
    ])
    return transform


def generate_negative_samples(img, mask):
    """
    Generate negative samples by creating regions without vessels.

    Inputs:
        img: Original image
        mask: Original mask

    Outputs:
        neg_img: Image with negative samples
        neg_mask: Mask for negative samples
    """
    # Create a binary mask where vessels are present
    vessel_present = mask > 0

    # Create a copy of the original image
    neg_img = img.copy()

    # Make label area white
    neg_img[vessel_present, :] = 255

    # Create an empty mask (all zeros)
    neg_mask = np.zeros_like(mask)

    return neg_img, neg_mask


def generate_artificial_vessels(img, mask):
    """
    Generate artificial vessel samples by warping and transforming existing vessels.

    Inputs:
        img: Original image
        mask: Original mask

    Outputs:
        art_img: Image with artificial vessels
        art_mask: Mask for artificial vessels
    """
    # Check if there are any vessels in the mask
    if np.max(mask) == 0:
        return None, None

    # Create copies of the original image and mask
    art_img = img.copy()

    img_len = img.shape[0]

    # Find vessel regions
    vessel_regions = mask > 0

    if np.sum(vessel_regions) > 0:
        # Extract vessel pixels
        vessel_img = np.zeros_like(img)
        vessel_img[vessel_regions] = img[vessel_regions]

        # Apply random transformations
        # 1. Random rotation
        angle = np.random.uniform(-180, 180)
        rows, cols = vessel_img.shape[:2]
        M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)

        # 2. Random scaling
        scale = np.random.uniform(0.25, 3)
        M[:, 0:2] = M[:, 0:2] * scale

        # 3. Random translation
        translation_factor = 0.5
        tx = np.random.randint(-img_len * translation_factor,
                               img_len * translation_factor)
        ty = np.random.randint(-img_len * translation_factor,
                               img_len * translation_factor)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply transformations
        warped_vessel = cv.warpAffine(
            vessel_img, M, (cols, rows), borderMode=cv.BORDER_REFLECT)
        warped_mask = cv.warpAffine(mask.astype(
            np.uint8), M, (cols, rows), borderMode=cv.BORDER_REFLECT)

        # Warping introduces variability, so assign value to closest in original
        orig_mask_values = np.unique(mask)
        warped_mask_list = []
        for val in orig_mask_values:
            this_mask = warped_mask.copy()
            this_mask[warped_mask != val] = 0
            this_mask = this_mask.astype(float)
            val = val.astype(float)
            max_orig = np.max(orig_mask_values).astype(float)
            this_mask = (this_mask / val) * (val + max_orig + 1)
            warped_mask_list.append(this_mask)
        warped_mask = np.nansum(warped_mask_list, axis=0)
        warped_mask = warped_mask.astype(np.uint16)

        fin_mask = mask.copy()
        fin_mask = fin_mask.astype(np.uint16)
        fin_mask[warped_mask > 0] = warped_mask[warped_mask > 0]

        # Renormalize
        fin_mask = fin_mask / np.max(fin_mask) * 255
        art_mask = fin_mask.astype(np.uint8)

        # Combine the original image with the warped vessel
        art_img = img.copy()
        art_img[warped_mask > 0] = warped_vessel[warped_mask > 0]

    return art_img, art_mask


def augment_dataset(images, masks, neg_ratio=0.2, art_ratio=0.5):
    """
    Augment a dataset with negative samples and artificial vessels.

    Inputs:
        images: List of original images
        masks: List of original masks
        neg_ratio: Ratio of negative samples to add
        art_ratio: Ratio of artificial vessel samples to add

    Outputs:
        aug_images: List of augmented images
        aug_masks: List of augmented masks
    """
    print(
        f'Starting dataset augmentation with {len(images)} original images...')
    aug_images = images.copy()
    aug_masks = masks.copy()

    num_orig = len(images)
    num_neg = int(num_orig * neg_ratio)
    num_art = int(num_orig * art_ratio)

    print(
        f'Will generate {num_neg} negative samples and {num_art} artificial vessel samples')

    # Generate negative samples
    print('Generating negative samples...')
    for i in trange(min(num_neg, num_orig)):
        neg_imgs, neg_msks = generate_negative_samples(images[i], masks[i])
        aug_images.append(neg_imgs)
        aug_masks.append(neg_msks)

    # Generate artificial vessel samples
    print('Generating artificial vessel samples...')
    for i in trange(min(num_art, num_orig)):
        art_imgs, art_msks = generate_artificial_vessels(images[i], masks[i])
        aug_images.append(art_imgs)
        aug_masks.append(art_msks)

    print(
        f'Dataset augmentation complete. Total images: {len(aug_images)} (original: {num_orig}, negative: {num_neg}, artificial: {num_art})')
    return aug_images, aug_masks


def load_or_create_augmented_data(
        labelled_data_dir,
        img_dir,
        mask_dir,
        train_imgs,
        train_masks,
):
    """
    Load existing augmented data or create a new augmented dataset.

    If augmented data already exists, it loads the filenames.
    Otherwise, it creates a new augmented dataset by:
    1. Loading a subset of training images
    2. Applying augmentations (negative samples, artificial vessels)
    3. Saving the augmented images and masks to disk

    Args:
        labelled_data_dir (str): Directory containing labelled data
        img_dir (str): Directory containing original images
        mask_dir (str): Directory containing original masks
        train_imgs (list): List of training image filenames
        train_masks (list): List of training mask filenames

    Returns:
        tuple: (aug_img_dir, aug_mask_dir, aug_img_names, aug_mask_names) -
               Directories and lists of augmented image and mask filenames
    """
    print('Checking for existing augmented data...')
    aug_img_dir = os.path.join(labelled_data_dir, 'augmented_images/')
    aug_mask_dir = os.path.join(labelled_data_dir, 'augmented_masks/')

    if os.path.exists(aug_img_dir) and os.path.exists(aug_mask_dir) \
            and len(os.listdir(aug_img_dir)) > 0 and len(os.listdir(aug_mask_dir)) > 0:
        print("Augmented images already exist. Skipping augmentation...")
        aug_img_names = sorted(os.listdir(aug_img_dir))
        aug_mask_names = sorted(os.listdir(aug_mask_dir))
        print(f'Loaded {len(aug_img_names)} existing augmented images')
    else:
        # Create augmented dataset paths
        print("Creating augmented dataset...")
        n_augmented = len(train_imgs) * 10
        print(
            f'Will create {n_augmented} augmented samples from {len(train_imgs)} training images')

        # Load a few images to augment
        print('Loading base images for augmentation...')
        aug_img_list = []
        aug_mask_list = []
        for i in np.random.choice(range(len(train_imgs)), n_augmented, replace=True):
            img = np.array(Image.open(img_dir + train_imgs[i]).convert("RGB"))
            mask = np.array(Image.open(mask_dir + train_masks[i]))
            aug_img_list.append(img)
            aug_mask_list.append(mask)

        # Augment the dataset
        aug_images, aug_masks = augment_dataset(
            aug_img_list, aug_mask_list, neg_ratio=0.3, art_ratio=0.5)

        # Save augmented images and masks
        print('Saving augmented images to disk...')
        os.makedirs(aug_img_dir, exist_ok=True)
        os.makedirs(aug_mask_dir, exist_ok=True)

        aug_img_names = []
        aug_mask_names = []
        for i, (img, mask) in enumerate(tqdm(zip(aug_images, aug_masks), desc='Saving augmented data')):
            img_name = f'aug_{i:03}.png'
            mask_name = f'aug_{i:03}_mask.png'

            # Save the augmented image and mask
            plt.imsave(os.path.join(aug_img_dir, img_name), img)
            plt.imsave(os.path.join(aug_mask_dir, mask_name),
                       mask, cmap='gray')

            aug_img_names.append(img_name)
            aug_mask_names.append(mask_name)

        print(
            f'Successfully created and saved {len(aug_img_names)} augmented images')

    # Validate augmented images
    print('Validating augmented image-mask pairs...')
    for img_name, mask_name in zip(aug_img_names, aug_mask_names):
        assert img_name.split(".")[0] in mask_name
    print('Validation complete - all image-mask pairs match')

    return aug_img_dir, aug_mask_dir, aug_img_names, aug_mask_names


def combine_datasets(
        train_imgs,
        train_masks,
        val_imgs,
        val_masks,
        aug_img_names,
        aug_mask_names,
):
    """
    Combine original and augmented datasets.

    Merges the original dataset with augmented samples,
    maintaining the train/validation split for all data types.

    Args:
        train_imgs (list): List of original training image filenames
        train_masks (list): List of original training mask filenames
        val_imgs (list): List of original validation image filenames
        val_masks (list): List of original validation mask filenames
        aug_img_names (list): List of augmented image filenames
        aug_mask_names (list): List of augmented mask filenames

    Returns:
        tuple: (train_imgs, train_masks, val_imgs, val_masks) -
               Lists of combined filenames for training and validation
    """
    print('Combining datasets...')
    print(f'Original - Train: {len(train_imgs)}, Val: {len(val_imgs)}')
    print(f'Augmented: {len(aug_img_names)}')

    n_aug_train = int(0.9 * len(aug_img_names))

    print(
        f'Splitting augmented data - Train: {n_aug_train}, Val: {len(aug_img_names) - n_aug_train}')

    # Combine datasets
    train_imgs = np.concatenate(
        [train_imgs,
         aug_img_names[:n_aug_train],
         ]
    )
    train_masks = np.concatenate(
        [
            train_masks,
            aug_mask_names[:n_aug_train],
        ]
    )
    val_imgs = np.concatenate(
        [
            val_imgs,
            aug_img_names[n_aug_train:],
        ]
    )
    val_masks = np.concatenate(
        [
            val_masks,
            aug_mask_names[n_aug_train:],
        ]
    )

    print(
        f'Final combined dataset - Train: {len(train_imgs)}, Val: {len(val_imgs)}')

    # Check that all images have corresponding masks
    print('Validating combined dataset image-mask pairs...')
    for img_name, mask_name in zip(train_imgs, train_masks):
        assert img_name.split(".")[0] in mask_name
    for img_name, mask_name in zip(val_imgs, val_masks):
        assert img_name.split(".")[0] in mask_name
    print('Dataset combination and validation complete')

    return train_imgs, train_masks, val_imgs, val_masks


#############################################################################
# Dataset Classes
#############################################################################

class AugmentedCustDat(torch.utils.data.Dataset):
    """
    Dataset class that can handle original, augmented, and negative images.

    This extended dataset class supports loading images from multiple directories
    (original, augmented, and negative) and applies appropriate transformations.
    It also handles the conversion of masks to the format required by Mask R-CNN.
    """

    def __init__(self, image_names, mask_names, img_dir, mask_dir,
                 aug_img_dir, aug_mask_dir,
                 transform=None):
        self.image_names = image_names
        self.mask_names = mask_names
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.aug_img_dir = aug_img_dir
        self.aug_mask_dir = aug_mask_dir
        self.base_transform = T.ToTensor()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.base_transform

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = self.mask_names[idx]

        # Check if this is an augmented image
        if 'aug_' in img_name:
            img = Image.open(os.path.join(
                self.aug_img_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(self.aug_mask_dir, mask_name))
        else:
            img = Image.open(self.img_dir + img_name).convert("RGB")
            mask = Image.open(self.mask_dir + mask_name)

        # Apply transformations
        img_tensor, mask_tensor = self.transform(img, mask)

        # Convert mask back to numpy array
        mask = mask_tensor.numpy()[0] * 255
        mask = mask.astype(np.uint8)

        # Filter objects by size
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        fin_objs = []
        for obj in obj_ids:
            if np.mean(mask == obj) > 0.005:
                fin_objs.append(obj)

        obj_ids = np.array(fin_objs)

        num_objs = len(obj_ids)
        masks = np.zeros((num_objs, mask.shape[0], mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == obj_ids[i]] = True

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i] > 0)
            if len(pos[0]) == 0:  # Skip if mask is empty
                continue
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:  # If no valid boxes, create a dummy box
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros(
                (0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        return img_tensor, target

    def __len__(self):
        return len(self.image_names)


def custom_collate(data):
    """
    Custom collate function for DataLoader.

    This function is used to handle variable-sized images and masks
    in the batch without padding or resizing.

    Args:
        data: Batch of data from the dataset

    Returns:
        The same data without modification
    """
    return data


def create_dataloaders(train_imgs, train_masks, val_imgs, val_masks,
                       img_dir, mask_dir, aug_img_dir, aug_mask_dir,
                       transform):
    """
    Create DataLoader objects for training and validation.

    Sets up PyTorch DataLoaders with appropriate batch size, shuffling,
    and other parameters for efficient training and validation.

    Args:
        train_imgs (list): List of training image filenames
        train_masks (list): List of training mask filenames
        val_imgs (list): List of validation image filenames
        val_masks (list): List of validation mask filenames
        img_dir (str): Directory containing original images
        mask_dir (str): Directory containing original masks
        aug_img_dir (str): Directory containing augmented images
        aug_mask_dir (str): Directory containing augmented masks
        transform (callable): Transformation function to apply to the data

    Returns:
        tuple: (train_dl, val_dl) - DataLoader objects for training and validation
    """
    print('Creating DataLoaders...')
    batch_size = 2
    num_workers = 1
    use_cuda = torch.cuda.is_available()

    print(f'DataLoader configuration:')
    print(f'  Batch size: {batch_size}')
    print(f'  Num workers: {num_workers}')
    print(f'  CUDA available: {use_cuda}')
    print(f'  Pin memory: {use_cuda}')

    train_dl = torch.utils.data.DataLoader(
        AugmentedCustDat(
            train_imgs, train_masks,
            img_dir, mask_dir,
            aug_img_dir, aug_mask_dir,
            transform
        ),
        batch_size=batch_size,
        shuffle=True,  # Changed to True for better training
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True
    )

    val_dl = torch.utils.data.DataLoader(
        AugmentedCustDat(
            val_imgs, val_masks,
            img_dir, mask_dir,
            aug_img_dir, aug_mask_dir,
            transform
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True
    )

    print(
        f'DataLoaders created - Train batches: {len(train_dl)}, Val batches: {len(val_dl)}')
    return train_dl, val_dl


#############################################################################
# Visualization Functions
#############################################################################

def plot_augmented_samples(aug_img_dir, aug_mask_dir, aug_img_names, aug_mask_names, plot_dir):
    """
    Plot a grid of randomly selected augmented images with their masks.

    Creates a visualization of augmented samples to verify the quality
    and variety of the augmented dataset.

    Args:
        aug_img_dir (str): Directory containing augmented images
        aug_mask_dir (str): Directory containing augmented masks
        aug_img_names (list): List of augmented image filenames
        aug_mask_names (list): List of augmented mask filenames
        plot_dir (str): Directory to save the visualization
    """
    n_plot = 25
    fig, ax = plt.subplots(
        np.ceil(np.sqrt(n_plot)).astype(int),
        np.ceil(np.sqrt(n_plot)).astype(int),
        figsize=(15, 15))
    for i in range(n_plot):
        rand_ind = np.random.choice(range(len(aug_img_names)))
        img_name = aug_img_names[rand_ind]
        mask_name = aug_mask_names[rand_ind]
        img = Image.open(aug_img_dir + img_name).convert("RGB")
        mask = Image.open(aug_mask_dir + mask_name).convert("L")
        # Get outline of mask
        ax.flatten()[i].imshow(img)
        mask_array = np.array(mask)
        if np.sum(mask_array) > 0:
            mask_outline = get_mask_outline(np.array(mask) > 0)
            ax.flatten()[i].scatter(*np.where(mask_outline)[::-1], c='y', s=1)
        ax.flatten()[i].axis('off')
        ax.flatten()[i].set_title(img_name + '\n' + mask_name)

    fig.savefig(plot_dir + '/augmented_images.png')
    plt.close(fig)

    print(
        f'Plotted {n_plot} augmented samples to {plot_dir}/augmented_images.png')


def create_sample_plots(train_imgs, train_masks, val_imgs, val_masks,
                        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
                        plot_dir):
    """
    Create sample plots of training and validation images with their masks.

    Randomly selects samples from the training and validation sets and
    creates visualizations to verify the data distribution.

    Args:
        train_imgs (list): List of training image filenames
        train_masks (list): List of training mask filenames
        val_imgs (list): List of validation image filenames
        val_masks (list): List of validation mask filenames
        img_dir (str): Directory containing original images
        mask_dir (str): Directory containing original masks
        aug_img_dir (str): Directory containing augmented images
        aug_mask_dir (str): Directory containing augmented masks
        plot_dir (str): Directory to save the visualizations
    """
    test_plot_dir = os.path.join(plot_dir, 'train_val_split')
    os.makedirs(test_plot_dir, exist_ok=True)

    n_plot = 10
    # Plot training samples
    train_inds = np.random.choice(
        range(len(train_imgs)), n_plot, replace=False)
    for img_name, mask_name in zip(train_imgs[train_inds], train_masks[train_inds]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if 'aug_' in img_name:
            img = Image.open(aug_img_dir + img_name).convert("RGB")
            mask = Image.open(aug_mask_dir + mask_name)
        else:
            img = Image.open(img_dir + img_name).convert("RGB")
            mask = Image.open(mask_dir + mask_name)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        fig.savefig(test_plot_dir + f'/{img_name.split(".")[0]}train.png')
        plt.close(fig)

    # Plot validation samples
    val_inds = np.random.choice(range(len(val_imgs)), n_plot, replace=False)
    for img_name, mask_name in zip(val_imgs[val_inds], val_masks[val_inds]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if 'aug_' in img_name:
            img = Image.open(aug_img_dir + img_name).convert("RGB")
            mask = Image.open(aug_mask_dir + mask_name)
        else:
            img = Image.open(img_dir + img_name).convert("RGB")
            mask = Image.open(mask_dir + mask_name)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        fig.savefig(test_plot_dir + f'/{img_name.split(".")[0]}val.png')
        plt.close(fig)

    print(
        f'Created sample plots for training and validation datasets in {test_plot_dir}')


def test_transformations(img_dir, mask_dir, image_names, mask_names, transform):
    """
    Test the transformations on a sample image and visualize the results.

    This function tests and visualizes:
    1. Basic transformations (flips, rotations, color jitter)
    2. Negative sample generation
    3. Artificial vessel generation

    Args:
        img_dir (str): Directory containing original images
        mask_dir (str): Directory containing original masks
        image_names (list): List of image filenames
        mask_names (list): List of mask filenames
        transform (callable): Transformation function to test
    """
    idx = 4
    img = Image.open(img_dir + image_names[idx]).convert("RGB")
    mask = Image.open(mask_dir + mask_names[idx])

    # Test transform
    img_transformed, mask_transformed = transform(img, mask)

    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].imshow(img_transformed.T)
    axis[1].imshow(mask_transformed.T)
    plt.show()

    # Test negative image generation
    img_np = np.array(img)
    mask_np = np.array(mask)
    neg_img, neg_mask = generate_negative_samples(img_np, mask_np)

    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].imshow(neg_img)
    axis[1].imshow(neg_mask)
    plt.show()

    # Test artificial vessel generation
    art_img, art_mask = generate_artificial_vessels(img_np, mask_np)
    art_mask_outline = get_mask_outline(art_mask)

    fig, axis = plt.subplots(2, 2, figsize=(10, 10))
    axis[0, 0].imshow(img_np)
    axis[0, 1].imshow(mask_np)
    axis[1, 0].imshow(art_img)
    axis[1, 1].imshow(art_mask)
    axis[1, 0].scatter(*np.where(art_mask_outline)[::-1], c='y', s=1)
    plt.show()


#############################################################################
# Main Preprocessing Pipeline
#############################################################################

def prepare_data(data_dir=None, use_augmentation=True):
    """
    Main data preprocessing pipeline.

    Args:
        data_dir (str): Root data directory
        use_augmentation (bool): Whether to use data augmentation

    Returns:
        tuple: All necessary data components for training
    """
    print("Starting data preprocessing pipeline...")

    # Load original data
    labelled_data_dir, img_dir, mask_dir, image_names, mask_names = load_data(
        data_dir)

    # Create transforms
    transform = create_transforms()

    # Split data
    train_imgs, train_masks, val_imgs, val_masks = split_train_val(
        image_names, mask_names)

    if use_augmentation:
        # Load or create augmented data
        aug_img_dir, aug_mask_dir, aug_img_names, aug_mask_names = load_or_create_augmented_data(
            labelled_data_dir, img_dir, mask_dir, train_imgs, train_masks
        )

        # Combine datasets
        train_imgs, train_masks, val_imgs, val_masks = combine_datasets(
            train_imgs, train_masks, val_imgs, val_masks,
            aug_img_names, aug_mask_names,
        )
    else:
        # Use empty augmented directories if not using augmentation
        aug_img_dir = os.path.join(labelled_data_dir, 'augmented_images/')
        aug_mask_dir = os.path.join(labelled_data_dir, 'augmented_masks/')
        os.makedirs(aug_img_dir, exist_ok=True)
        os.makedirs(aug_mask_dir, exist_ok=True)
        aug_img_names = []
        aug_mask_names = []

    # Create dataloaders
    train_dl, val_dl = create_dataloaders(
        train_imgs, train_masks, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        transform
    )

    print("Data preprocessing pipeline complete!")

    return {
        'train_dl': train_dl,
        'val_dl': val_dl,
        'train_imgs': train_imgs,
        'train_masks': train_masks,
        'val_imgs': val_imgs,
        'val_masks': val_masks,
        'img_dir': img_dir,
        'mask_dir': mask_dir,
        'aug_img_dir': aug_img_dir,
        'aug_mask_dir': aug_mask_dir,
        'transform': transform,
        'labelled_data_dir': labelled_data_dir,
        'aug_img_names': aug_img_names,
        'aug_mask_names': aug_mask_names
    }
