"""
Utilities for UNet model training and prediction.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from autoslide.src.pipeline.model.unet_model import UNet


class UNetDataset(Dataset):
    """Dataset for UNet training"""
    
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


def initialize_unet_model(n_channels=3, n_classes=1, bilinear=False):
    """
    Initialize UNet model.
    
    Args:
        n_channels (int): Number of input channels (3 for RGB)
        n_classes (int): Number of output classes (1 for binary segmentation)
        bilinear (bool): Use bilinear upsampling instead of transposed convolutions
    
    Returns:
        UNet: Initialized UNet model
    """
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    return model


def create_unet_dataloaders(train_imgs, train_masks, val_imgs, val_masks, 
                            img_dir, mask_dir, batch_size=4):
    """
    Create DataLoaders for UNet training.
    
    Args:
        train_imgs (list): Training image filenames
        train_masks (list): Training mask filenames
        val_imgs (list): Validation image filenames
        val_masks (list): Validation mask filenames
        img_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        batch_size (int): Batch size for training
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    train_img_paths = [os.path.join(img_dir, name) for name in train_imgs]
    train_mask_paths = [os.path.join(mask_dir, name) for name in train_masks]
    val_img_paths = [os.path.join(img_dir, name) for name in val_imgs]
    val_mask_paths = [os.path.join(mask_dir, name) for name in val_masks]
    
    train_dataset = UNetDataset(train_img_paths, train_mask_paths, transform=transform)
    val_dataset = UNetDataset(val_img_paths, val_mask_paths, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def train_unet_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train UNet for one epoch.
    
    Args:
        model (nn.Module): UNet model
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_unet(model, dataloader, criterion, device):
    """
    Validate UNet model.
    
    Args:
        model (nn.Module): UNet model
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def predict_unet(model, image, device, transform=None, threshold=0.5):
    """
    Perform prediction with UNet model.
    
    Args:
        model (nn.Module): Trained UNet model
        image (PIL.Image or str): Input image or path to image
        device: Device to run inference on
        transform: Image transformation function
        threshold (float): Threshold for binary mask
    
    Returns:
        numpy.ndarray: Predicted mask (0-255)
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    if transform is None:
        transform = T.ToTensor()
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        # Apply threshold
        mask = (probs > threshold).float()
    
    # Convert to numpy and scale to 0-255
    mask = mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    
    return mask
