"""
Common utilities for model prediction and evaluation.

This module provides shared functionality for:
- Model loading and initialization
- Single image prediction
- Common imports and setup
- Support for both Mask R-CNN and UNet models
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from autoslide.src import config
from autoslide.src.pipeline.model.unet_model import UNet

# Get directories from config - use project-level artifacts directory
# Navigate from autoslide/src/pipeline/model/ to autoslide/artifacts/
artifacts_dir = os.path.join(
    os.path.dirname(__file__),  # autoslide/src/pipeline/model/
    '..', '..', '..', '..',      # up to project root
    'autoslide', 'artifacts'     # down to autoslide/artifacts/
)
artifacts_dir = os.path.abspath(artifacts_dir)


def initialize_model(model_type='maskrcnn'):
    """
    Initialize and configure the segmentation model.

    Args:
        model_type (str): Type of model to initialize ('maskrcnn' or 'unet')

    Returns:
        torch.nn.Module: Configured model
    """
    if model_type.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=1, bilinear=False)
        return model
    elif model_type.lower() == 'maskrcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()

        # Configure for binary classification (background + vessel)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)

        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'maskrcnn' or 'unet'.")


def load_model(model_path=None, device=None, model_type='maskrcnn'):
    """
    Load the trained segmentation model.

    Args:
        model_path (str): Path to the saved model. If None, uses default path.
        device (torch.device): Device to load model on. If None, auto-detects.
        model_type (str): Type of model to load ('maskrcnn' or 'unet')

    Returns:
        tuple: (model, device, transform) - Loaded model, device, and transform
    """
    if model_path is None:
        if model_type.lower() == 'unet':
            model_path = os.path.join(artifacts_dir, 'best_val_unet_model.pth')
        else:
            model_path = os.path.join(artifacts_dir, 'best_val_mask_rcnn_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure
    model = initialize_model(model_type=model_type)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform = T.ToTensor()

    return model, device, transform


def predict_single_image(model, image, device, transform, return_time=False, model_type='maskrcnn', threshold=0.5):
    """
    Perform prediction on a single image.

    Args:
        model (torch.nn.Module): Trained segmentation model
        image (PIL.Image or str): Input image or path to image
        device (torch.device): Device to run inference on
        transform (callable): Image transformation function
        return_time (bool): Whether to return prediction time
        model_type (str): Type of model ('maskrcnn' or 'unet')
        threshold (float): Threshold for UNet predictions

    Returns:
        numpy.ndarray or tuple: Combined predicted mask, optionally with prediction time
    """
    # Handle both PIL Image and file path inputs
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL.Image or file path string")

    # Transform image
    img_tensor = transform(image).to(device)

    # Measure prediction time if requested
    start_time = time.time() if return_time else None

    with torch.no_grad():
        if model_type.lower() == 'unet':
            # UNet prediction
            output = model(img_tensor.unsqueeze(0))
            probs = torch.sigmoid(output)
            mask = (probs > threshold).float()
            combined_mask = mask.squeeze().cpu().numpy()
            combined_mask = (combined_mask * 255).astype(np.uint8)
        else:
            # Mask R-CNN prediction
            predictions = model([img_tensor])
            pred = predictions[0]
            
            if len(pred["masks"]) > 0:
                # Combine all masks with confidence weighting
                masks = pred["masks"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()

                # Weight masks by their confidence scores
                combined_mask = np.zeros_like(masks[0, 0])
                total_weight = 0

                for mask, score in zip(masks, scores):
                    combined_mask += mask[0] * score
                    total_weight += score

                # Normalize to 0-1 range, then convert to 0-255
                if total_weight > 0:
                    combined_mask = combined_mask / total_weight

                if combined_mask.max() > 0:
                    combined_mask = combined_mask / combined_mask.max()

                # Convert to uint8 (0-255 range)
                combined_mask = (combined_mask * 255).astype(np.uint8)
            else:
                # No predictions - create empty mask
                img_array = np.array(image)
                combined_mask = np.zeros(
                    (img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    end_time = time.time() if return_time else None
    prediction_time = (end_time - start_time) if return_time else None

    if return_time:
        return prediction_time, combined_mask
    else:
        return combined_mask


def setup_device():
    """
    Setup and return the appropriate device for inference.

    Returns:
        torch.device: Device to use for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
