import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm, trange
import torchvision

#############################################################################
# Directory and Data Management Functions
#############################################################################

def setup_directories(autoslide_dir):
    """
    Set up necessary directories for saving artifacts and plots.
    
    Args:
        autoslide_dir (str): Root directory of the AutoSlide project
        
    Returns:
        tuple: (plot_dir, artifacts_dir) - Paths to the plot and artifacts directories
    """
    plot_dir = os.path.join(autoslide_dir, 'plots') 
    artifacts_dir = os.path.join(autoslide_dir, 'artifacts')
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    return plot_dir, artifacts_dir

def load_data(autoslide_dir):
    """
    Load original image and mask data from the labelled_images directory.
    
    Args:
        autoslide_dir (str): Root directory of the AutoSlide project
        
    Returns:
        tuple: (labelled_data_dir, img_dir, mask_dir, image_names, mask_names) - 
               Directories and lists of image and mask filenames
    """
    labelled_data_dir = os.path.join(autoslide_dir, 'data/labelled_images')
    img_dir = os.path.join(labelled_data_dir, 'images/') 
    mask_dir = os.path.join(labelled_data_dir, 'masks/') 
    image_names = sorted(os.listdir(img_dir))
    mask_names = sorted(os.listdir(mask_dir))
    
    # Validate that masks correspond to images
    for img_path, mask_path in zip(image_names, mask_names):
        assert img_path.split(".")[0] in mask_path
        
    return labelled_data_dir, img_dir, mask_dir, image_names, mask_names

def get_mask_outline(mask):
    """
    Extract the outline of a mask for visualization purposes.
    
    Args:
        mask (numpy.ndarray): Binary mask image
        
    Returns:
        numpy.ndarray: Outline of the mask as a binary image
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_outline = np.zeros_like(mask)
    mask_outline = cv2.drawContours(mask_outline, contours, -1, 255, 1)
    return mask_outline

#############################################################################
# Data Augmentation and Transformation Functions
#############################################################################

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
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor()
            ])
    return transform

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
    aug_images = images.copy()
    aug_masks = masks.copy()
    
    num_orig = len(images)
    num_neg = int(num_orig * neg_ratio)
    num_art = int(num_orig * art_ratio)
    
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
    
    return aug_images, aug_masks

def gen_negative_masks(neg_dir, output_dir):
    neg_image_names = os.listdir(neg_dir)
    out_mask_names = [f'{name.split(".")[0]}_mask.png' for name in neg_image_names]
    for img_name, mask_name in zip(neg_image_names, out_mask_names):
        img_path = os.path.join(neg_dir, img_name)
        mask_path = os.path.join(output_dir, mask_name)
        img = plt.imread(img_path)
        mask = np.zeros_like(img)
        plt.imsave(mask_path, mask)

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
        # x_scale = np.random.uniform(0.8, 1.2)
        # y_scale = np.random.uniform(0.8, 1.2)
        # M[:, 0] = M[:, 0] * x_scale
        # M[:, 1] = M[:, 1] * y_scale
        
        # 3. Random translation
        translation_factor = 0.5
        tx = np.random.randint(-img_len * translation_factor, img_len * translation_factor)
        ty = np.random.randint(-img_len * translation_factor, img_len * translation_factor)
        # tx = np.random.randint(-20, 20)
        # ty = np.random.randint(-20, 20)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformations
        warped_vessel = cv.warpAffine(vessel_img, M, (cols, rows), borderMode=cv.BORDER_REFLECT)
        warped_mask = cv.warpAffine(mask.astype(np.uint8), M, (cols, rows), borderMode=cv.BORDER_REFLECT)
        
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

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(mask)
        # ax[1].imshow(warped_mask)
        # ax[2].imshow(fin_mask)
        # plt.show()

        # # Renormalize mask
        # warped_mask = warped_mask / np.max(warped_mask) * 255
        #
        # warped_mask = (warped_mask > 0).astype(np.uint8) * 255
        # 
        # # Place the warped vessel in a different location
        # # Find non-vessel regions in the original image
        # non_vessel = ~vessel_regions
        # 
        # # Create a new mask for the artificial vessel
        # art_mask = warped_mask
        # 
        # Combine the original image with the warped vessel
        art_img = img.copy()
        art_img[warped_mask > 0] = warped_vessel[warped_mask > 0]
        
    return art_img, art_mask 

#############################################################################
# Dataset Classes
#############################################################################

class CustDat(torch.utils.data.Dataset):
    """
    Dataset class for the original dataset without augmentations.
    
    This class handles loading images and masks, applying transformations,
    and preparing the data in the format required by Mask R-CNN.
    """
    def __init__(self, image_names, mask_names, img_dir, mask_dir, transform=None):
        self.image_names = image_names
        self.mask_names = mask_names
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.base_transform = T.ToTensor()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.base_transform

    def __getitem__(self, idx):
        img = Image.open(self.img_dir + self.image_names[idx]).convert("RGB")
        mask = Image.open(self.mask_dir + self.mask_names[idx])

        # Apply transformations
        img_tensor, mask_tensor = self.transform(img, mask)

        # Convert mask back to numpy array
        mask = mask_tensor.numpy()[0] * 255
        mask = mask.astype(np.uint8)

        # Something weird is happening (likely with the transform)
        # 255 is showing up in a very small number of pixels
        # Enforce that a mask has to cover at least 1% of the image
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
            pos = np.where(masks[i]>0)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        return img_tensor, target

    def __len__(self):
        return len(self.image_names)

#############################################################################
# Model Initialization and Setup Functions
#############################################################################

def initialize_model():
    """
    Initialize and configure the Mask R-CNN model.
    
    Creates a Mask R-CNN model with ResNet-50 backbone and FPN,
    and configures it for binary segmentation (background and vessel).
    
    Returns:
        torchvision.models.detection.MaskRCNN: Configured Mask R-CNN model
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    return model

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

#############################################################################
# Dataset Preparation Functions
#############################################################################

def split_train_val(image_names, mask_names):
    """
    Split the dataset into training and validation sets.
    
    Randomly selects 90% of the data for training and 10% for validation,
    ensuring that the training set has an even number of samples.
    
    Args:
        image_names (list): List of image filenames
        mask_names (list): List of mask filenames
        
    Returns:
        tuple: (train_imgs, train_masks, val_imgs, val_masks) - 
               Lists of filenames for training and validation
    """
    num = int(0.9 * len(image_names))
    num = num if num % 2 == 0 else num + 1
    train_imgs_inds = np.random.choice(range(len(image_names)), num, replace=False)
    val_imgs_inds = np.setdiff1d(range(len(image_names)), train_imgs_inds)
    train_imgs = np.array(image_names)[train_imgs_inds]
    val_imgs = np.array(image_names)[val_imgs_inds]
    train_masks = np.array(mask_names)[train_imgs_inds]
    val_masks = np.array(mask_names)[val_imgs_inds]
    return train_imgs, train_masks, val_imgs, val_masks

def load_or_create_augmented_data(labelled_data_dir, img_dir, mask_dir, train_imgs, train_masks):
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
    aug_img_dir = os.path.join(labelled_data_dir, 'augmented_images/')
    aug_mask_dir = os.path.join(labelled_data_dir, 'augmented_masks/')
    
    if os.path.exists(aug_img_dir) and os.path.exists(aug_mask_dir) \
            and len(os.listdir(aug_img_dir)) > 0 and len(os.listdir(aug_mask_dir)) > 0:
        print("Augmented images already exist. Skipping augmentation...")
        aug_img_names = sorted(os.listdir(aug_img_dir))
        aug_mask_names = sorted(os.listdir(aug_mask_dir))
    else:
        # Create augmented dataset paths
        print("Creating augmented dataset...")
        n_augmented = len(train_imgs) * 10
        # Load a few images to augment
        aug_img_list = []
        aug_mask_list = []
        for i in np.random.choice(range(len(train_imgs)), n_augmented, replace=True):
            img = np.array(Image.open(img_dir + train_imgs[i]).convert("RGB"))
            mask = np.array(Image.open(mask_dir + train_masks[i]))
            aug_img_list.append(img)
            aug_mask_list.append(mask)

        # Augment the dataset
        aug_images, aug_masks = augment_dataset(aug_img_list, aug_mask_list, neg_ratio=0.3, art_ratio=0.5)

        # Save augmented images and masks
        os.makedirs(aug_img_dir, exist_ok=True)
        os.makedirs(aug_mask_dir, exist_ok=True)

        aug_img_names = []
        aug_mask_names = []
        for i, (img, mask) in enumerate(tqdm(zip(aug_images, aug_masks))):
            img_name = f'aug_{i:03}.png'
            mask_name = f'aug_{i:03}_mask.png'
            
            # Save the augmented image and mask
            plt.imsave(os.path.join(aug_img_dir, img_name), img)
            plt.imsave(os.path.join(aug_mask_dir, mask_name), mask, cmap='gray')
            
            aug_img_names.append(img_name)
            aug_mask_names.append(mask_name)

    # Validate augmented images
    for img_name, mask_name in zip(aug_img_names, aug_mask_names):
        assert img_name.split(".")[0] in mask_name
        
    return aug_img_dir, aug_mask_dir, aug_img_names, aug_mask_names

def load_negative_images(labelled_data_dir):
    """
    Load negative images (images without vessels).
    
    Negative images are used to help the model learn to distinguish
    between regions with and without vessels.
    
    Args:
        labelled_data_dir (str): Directory containing labelled data
        
    Returns:
        tuple: (neg_image_dir, neg_mask_dir, neg_img_names, neg_mask_names) -
               Directories and lists of negative image and mask filenames
    """
    neg_image_dir = os.path.join(labelled_data_dir, 'negative_images/')
    neg_mask_dir = os.path.join(labelled_data_dir, 'negative_masks/')
    neg_img_names = sorted(os.listdir(neg_image_dir))
    neg_mask_names = sorted(os.listdir(neg_mask_dir))
    
    print(f'Negative images: {len(np.unique(neg_img_names))}')
    
    return neg_image_dir, neg_mask_dir, neg_img_names, neg_mask_names

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
            mask_outline = get_mask_outline(np.array(mask)>0)
            ax.flatten()[i].scatter(*np.where(mask_outline)[::-1], c='y', s=1)
        ax.flatten()[i].axis('off')
        ax.flatten()[i].set_title(img_name + '\n' + mask_name)
    
    fig.savefig(plot_dir + '/augmented_images.png')
    plt.close(fig)

def combine_datasets(train_imgs, train_masks, val_imgs, val_masks, 
                    aug_img_names, aug_mask_names, neg_img_names, neg_mask_names):
    """
    Combine original, augmented, and negative datasets.
    
    Merges the original dataset with augmented and negative samples,
    maintaining the train/validation split for all data types.
    
    Args:
        train_imgs (list): List of original training image filenames
        train_masks (list): List of original training mask filenames
        val_imgs (list): List of original validation image filenames
        val_masks (list): List of original validation mask filenames
        aug_img_names (list): List of augmented image filenames
        aug_mask_names (list): List of augmented mask filenames
        neg_img_names (list): List of negative image filenames
        neg_mask_names (list): List of negative mask filenames
        
    Returns:
        tuple: (train_imgs, train_masks, val_imgs, val_masks) -
               Lists of combined filenames for training and validation
    """
    n_aug_train = int(0.9 * len(aug_img_names))
    n_neg_train = int(0.9 * len(neg_img_names))
    
    # Combine datasets
    train_imgs = np.concatenate([train_imgs, aug_img_names[:n_aug_train], neg_img_names[:n_neg_train]])
    train_masks = np.concatenate([train_masks, aug_mask_names[:n_aug_train], neg_mask_names[:n_neg_train]])
    val_imgs = np.concatenate([val_imgs, aug_img_names[n_aug_train:], neg_img_names[n_neg_train:]])
    val_masks = np.concatenate([val_masks, aug_mask_names[n_aug_train:], neg_mask_names[n_neg_train:]])
    
    # Check that all images have corresponding masks
    for img_name, mask_name in zip(train_imgs, train_masks):
        assert img_name.split(".")[0] in mask_name
    for img_name, mask_name in zip(val_imgs, val_masks):
        assert img_name.split(".")[0] in mask_name
        
    return train_imgs, train_masks, val_imgs, val_masks

def create_sample_plots(train_imgs, train_masks, val_imgs, val_masks, 
                       img_dir, mask_dir, aug_img_dir, aug_mask_dir, 
                       neg_image_dir, neg_mask_dir, plot_dir):
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
        neg_image_dir (str): Directory containing negative images
        neg_mask_dir (str): Directory containing negative masks
        plot_dir (str): Directory to save the visualizations
    """
    test_plot_dir = os.path.join(plot_dir, 'train_val_split')
    os.makedirs(test_plot_dir, exist_ok=True)


    n_plot = 10
    # Plot training samples
    train_inds = np.random.choice(range(len(train_imgs)), n_plot, replace=False)
    for img_name, mask_name in zip(train_imgs[train_inds], train_masks[train_inds]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if 'aug_' in img_name:
            img = Image.open(aug_img_dir + img_name).convert("RGB")
            mask = Image.open(aug_mask_dir + mask_name)
        elif 'neg_' in img_name:
            img = Image.open(neg_image_dir + img_name).convert("RGB")
            mask = Image.open(neg_mask_dir + mask_name)
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
        elif 'neg_' in img_name:
            img = Image.open(neg_image_dir + img_name).convert("RGB")
            mask = Image.open(neg_mask_dir + mask_name)
        else:
            img = Image.open(img_dir + img_name).convert("RGB")
            mask = Image.open(mask_dir + mask_name)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        fig.savefig(test_plot_dir + f'/{img_name.split(".")[0]}val.png')
        plt.close(fig)

class AugmentedCustDat(torch.utils.data.Dataset):
    """
    Dataset class that can handle original, augmented, and negative images.
    
    This extended dataset class supports loading images from multiple directories
    (original, augmented, and negative) and applies appropriate transformations.
    It also handles the conversion of masks to the format required by Mask R-CNN.
    """
    def __init__(self, image_names, mask_names, img_dir, mask_dir, 
                 aug_img_dir, aug_mask_dir, neg_image_dir, neg_mask_dir, transform=None):
        self.image_names = image_names
        self.mask_names = mask_names
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.aug_img_dir = aug_img_dir
        self.aug_mask_dir = aug_mask_dir
        self.neg_image_dir = neg_image_dir
        self.neg_mask_dir = neg_mask_dir
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
            img = Image.open(os.path.join(self.aug_img_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(self.aug_mask_dir, mask_name))
        elif 'neg_' in img_name:
            img = Image.open(os.path.join(self.neg_image_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(self.neg_mask_dir, mask_name))
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
            pos = np.where(masks[i]>0)
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
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
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

def create_dataloaders(train_imgs, train_masks, val_imgs, val_masks, 
                      img_dir, mask_dir, aug_img_dir, aug_mask_dir, 
                      neg_image_dir, neg_mask_dir, transform):
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
        neg_image_dir (str): Directory containing negative images
        neg_mask_dir (str): Directory containing negative masks
        transform (callable): Transformation function to apply to the data
        
    Returns:
        tuple: (train_dl, val_dl) - DataLoader objects for training and validation
    """
    train_dl = torch.utils.data.DataLoader(
        AugmentedCustDat(
            train_imgs, train_masks, 
            img_dir, mask_dir, 
            aug_img_dir, aug_mask_dir, 
            neg_image_dir, neg_mask_dir, 
            transform
        ),
        batch_size=2,
        shuffle=True,  # Changed to True for better training
        collate_fn=custom_collate,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_dl = torch.utils.data.DataLoader(
        AugmentedCustDat(
            val_imgs, val_masks, 
            img_dir, mask_dir, 
            aug_img_dir, aug_mask_dir, 
            neg_image_dir, neg_mask_dir, 
            transform
        ),
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    return train_dl, val_dl

#############################################################################
# Training and Evaluation Functions
#############################################################################

def setup_training(model, device):
    """
    Set up optimizer and other training parameters.
    
    Moves the model to the appropriate device (CPU/GPU) and
    configures the optimizer with appropriate learning rate and momentum.
    
    Args:
        model (torch.nn.Module): The Mask R-CNN model
        device (torch.device): Device to run the model on (CPU or GPU)
        
    Returns:
        torch.optim.Optimizer: Configured optimizer for training
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    return optimizer

def train_model(model, train_dl, val_dl, optimizer, device, plot_dir, artifacts_dir, n_epochs=90):
    """
    Train the model and evaluate on validation set.
    
    Performs the full training loop:
    1. Trains the model for the specified number of epochs
    2. Evaluates on the validation set after each epoch
    3. Tracks and saves the best model based on validation loss
    4. Plots training and validation losses
    5. Saves model checkpoints and loss histories
    
    Args:
        model (torch.nn.Module): The Mask R-CNN model
        train_dl (torch.utils.data.DataLoader): DataLoader for training data
        val_dl (torch.utils.data.DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Device to run the model on (CPU or GPU)
        plot_dir (str): Directory to save plots
        artifacts_dir (str): Directory to save model checkpoints
        n_epochs (int): Number of epochs to train for
        
    Returns:
        tuple: (model, all_train_losses, all_val_losses, best_val_loss) -
               Trained model, training losses, validation losses, and best validation loss
    """
    run_test_plot_dir = os.path.join(plot_dir, 'run_test_plot')
    os.makedirs(run_test_plot_dir, exist_ok=True)
    
    best_model_path = os.path.join(artifacts_dir, 'best_val_mask_rcnn_model.pth')
    fin_model_path = os.path.join(artifacts_dir, 'final_mask_rcnn_model.pth')
    
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')  # Track the best validation loss
    best_model = None
    flag = False
    
    for epoch in trange(n_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()
        n_train = len(train_dl)
        pbar = tqdm(train_dl)
        
        for i, dt in enumerate(pbar): 
            pbar.set_description(f"Epoch {epoch}/{n_epochs}, Batch {i}/{n_train}")
            imgs = [dt[0][0].to(device), dt[1][0].to(device)]
            targ = [dt[0][1], dt[1][1]]
            
            # Plot example image and mask to make sure augmentation is working
            if i == 0:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(imgs[0].cpu().detach().numpy().transpose(1, 2, 0))
                ax[1].imshow(np.sum(targ[0]['masks'].cpu().detach().numpy(), axis=0))
                # Plot boxes
                for box in targ[0]['boxes']:
                    ax[0].add_patch(
                            plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                          fill=False, color="y", linewidth=2))
                fig.savefig(run_test_plot_dir + f'/{epoch}_{i}_train.png')
                plt.close(fig)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs, targets)
            if not flag:
                print(loss)
                flag = True
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if np.isnan(train_epoch_loss):
                raise Exception('Loss is Nan')
                
        all_train_losses.append(train_epoch_loss)
        
        # Validation loop
        with torch.no_grad():
            for j, dt in enumerate(val_dl):
                if len(dt) < 2:
                    continue
                imgs = [dt[0][0].to(device), dt[1][0].to(device)]
                targ = [dt[0][1], dt[1][1]]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
                loss = model(imgs, targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()
            all_val_losses.append(val_epoch_loss)
            
        print(epoch, "  ", train_epoch_loss, "  ", val_epoch_loss)
        
        # Save model only if validation loss improves
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = model.state_dict()
            torch.save(best_model, best_model_path)

        torch.save(model.state_dict(), fin_model_path)

        # Plot training progress
        plot_losses(all_train_losses, all_val_losses, plot_dir, best_val_loss)
    
    # Save loss histories
    np.save(artifacts_dir + '/train_losses.npy', all_train_losses)
    np.save(artifacts_dir + '/val_losses.npy', all_val_losses)
    
    return model, all_train_losses, all_val_losses, best_val_loss

def plot_losses(all_train_losses, all_val_losses, plot_dir, best_val_loss):
    """
    Plot training and validation losses.
    
    Creates a visualization of training and validation loss curves
    with the best validation loss highlighted.
    
    Args:
        all_train_losses (list): List of training losses for each epoch
        all_val_losses (list): List of validation losses for each epoch
        plot_dir (str): Directory to save the plot
        best_val_loss (float): Best validation loss achieved during training
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(all_train_losses)
    ax[1].plot(all_val_losses)
    ax[0].set_title("Train Loss")
    ax[1].set_title("Validation Loss")
    ax[1].axhline(y=best_val_loss, color='r', linestyle='--', label='Best Val Loss')
    fig.savefig(plot_dir + '/train_val_loss.png')
    plt.close(fig)

def evaluate_model(model, val_imgs, val_masks, neg_img_names, neg_mask_names,
                  img_dir, mask_dir, aug_img_dir, aug_mask_dir, 
                  neg_image_dir, neg_mask_dir, device, plot_dir):
    """
    Evaluate the model on validation data and generate prediction visualizations.
    
    Runs the model on validation and negative samples and creates visualizations
    of the predictions compared to ground truth masks.
    
    Args:
        model (torch.nn.Module): The trained Mask R-CNN model
        val_imgs (list): List of validation image filenames
        val_masks (list): List of validation mask filenames
        neg_img_names (list): List of negative image filenames
        neg_mask_names (list): List of negative mask filenames
        img_dir (str): Directory containing original images
        mask_dir (str): Directory containing original masks
        aug_img_dir (str): Directory containing augmented images
        aug_mask_dir (str): Directory containing augmented masks
        neg_image_dir (str): Directory containing negative images
        neg_mask_dir (str): Directory containing negative masks
        device (torch.device): Device to run the model on (CPU or GPU)
        plot_dir (str): Directory to save the visualizations
    """
    model.eval()
    
    pred_out_path = os.path.join(plot_dir, 'pred_plots')
    os.makedirs(pred_out_path, exist_ok=True)
    
    transform = T.ToTensor()
    
    # Evaluate on validation set
    for img_name, mask_name in tqdm(zip(val_imgs, val_masks), total=len(val_imgs)):
        if 'aug_' in img_name:
            img = Image.open(aug_img_dir + img_name).convert("RGB")
            mask = Image.open(aug_mask_dir + mask_name)
        elif 'neg_' in img_name:
            img = Image.open(neg_image_dir + img_name).convert("RGB")
            mask = Image.open(neg_mask_dir + mask_name)
        else:
            img = Image.open(img_dir + img_name).convert("RGB")
            mask = Image.open(mask_dir + mask_name)

        # Use the base transform for prediction to match training
        ig = transform(img)
        with torch.no_grad():
            pred = model([ig.to(device)])

        n_preds = len(pred[0]["masks"])
        if n_preds > 0:
            fig, ax = plt.subplots(1, n_preds+1, figsize=(5*n_preds, 5))
            ax[0].imshow(img)
            for i in range(n_preds):
                ax[i+1].imshow((pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze())
            fig.savefig(pred_out_path + f'/{img_name.split(".")[0]}example_masks.png')
            plt.close(fig)

            all_preds = np.stack(
                    [
                        (pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze() \
                                for i in range(n_preds)
                                ]
                    )

            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(img)
            ax[1].imshow(all_preds.mean(axis=0))
            ax[2].imshow(np.array(mask))
            plt.savefig(pred_out_path + f'/{img_name.split(".")[0]}mean_example_mask.png')
            plt.close()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.imshow(img)
            ax.set_title('No predicted mask')
            plt.savefig(pred_out_path + f'/{img_name.split(".")[0]}_mean_example_mask.png')
            plt.close()
            print('No predicted mask')
    
    # Evaluate on negative examples
    for i, (img_name, mask_name) in enumerate(tqdm(zip(neg_img_names, neg_mask_names), total=len(neg_img_names))):
        print(img_name)
        if 'aug_' in img_name:
            img = Image.open(aug_img_dir + img_name).convert("RGB")
            mask = Image.open(aug_mask_dir + mask_name)
        elif 'neg_' in img_name:
            img = Image.open(neg_image_dir + img_name).convert("RGB")
            mask = Image.open(neg_mask_dir + mask_name)
        else:
            img = Image.open(img_dir + img_name).convert("RGB")
            mask = Image.open(mask_dir + mask_name)

        # Use the base transform for prediction to match training
        ig = transform(img)
        with torch.no_grad():
            pred = model([ig.to(device)])

        n_preds = len(pred[0]["masks"])
        if n_preds > 0:
            fig, ax = plt.subplots(1, n_preds+1, figsize=(5*n_preds, 5))
            ax[0].imshow(img)
            for i in range(n_preds):
                ax[i+1].imshow((pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze())
            fig.savefig(pred_out_path + f'/{img_name.split(".")[0]}_{i}_example_masks.png')
            plt.close(fig)

            all_preds = np.stack(
                    [
                        (pred[0]["masks"][i].cpu().detach().numpy() * 255).astype("uint8").squeeze() \
                                for i in range(n_preds)
                                ]
                    )

            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(img)
            ax[1].imshow(all_preds.mean(axis=0))
            ax[2].imshow(np.array(mask))
            plt.savefig(pred_out_path + f'/{img_name.split(".")[0]}_mean_example_mask.png')
            plt.close()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.imshow(img)
            ax.set_title('No predicted mask')
            plt.savefig(pred_out_path + f'/{img_name.split(".")[0]}_mean_example_mask.png')
            plt.close()
            print('No predicted mask')

def load_model(model, path, device):
    """
    Load a saved model from disk.
    
    Args:
        model (torch.nn.Module): The Mask R-CNN model structure
        path (str): Path to the saved model weights
        device (torch.device): Device to load the model on (CPU or GPU)
        
    Returns:
        torch.nn.Module: Loaded model with weights from disk
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model

