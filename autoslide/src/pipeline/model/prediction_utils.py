"""
Common utilities for model prediction and evaluation.

This module provides shared functionality for:
- Model loading and initialization
- Single image prediction
- Common imports and setup
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

# Get directories from config - use project-level artifacts directory
# Navigate from autoslide/src/pipeline/model/ to autoslide/artifacts/
artifacts_dir = os.path.join(
    os.path.dirname(__file__),  # autoslide/src/pipeline/model/
    '..', '..', '..', '..',      # up to project root
    'autoslide', 'artifacts'     # down to autoslide/artifacts/
)
artifacts_dir = os.path.abspath(artifacts_dir)


def initialize_model():
    """
    Initialize and configure the Mask R-CNN model.

    Creates a Mask R-CNN model with ResNet-50 backbone and FPN,
    and configures it for binary segmentation (background and vessel).

    Returns:
        torchvision.models.detection.MaskRCNN: Configured Mask R-CNN model
    """
    # COCO-pretrained backbone/FPN/RPN instead of random init (#121) - only
    # the box/mask predictor heads below are replaced for num_classes=2, so
    # this is evaluated in isolation from the v2 architecture / higher
    # mask_roi_pool resolution changes, which interact with the pretrained
    # mask head's expected 14x14 input and are tracked as separate
    # follow-ups rather than bundled in here.
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # Configure for binary classification (background + vessel)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model


def load_model(model_path=None, device=None):
    """
    Load the trained Mask R-CNN model.

    Args:
        model_path (str): Path to the saved model. If None, uses default path.
        device (torch.device): Device to load model on. If None, auto-detects.

    Returns:
        tuple: (model, device, transform) - Loaded model, device, and transform
    """
    if model_path is None:
        model_path = os.path.join(
            artifacts_dir, 'best_val_mask_rcnn_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure
    model = initialize_model()

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform = T.ToTensor()

    return model, device, transform


def combine_prediction_masks(masks, scores, mask_shape, score_threshold=0.3,
                             instance_mask_threshold=0.5):
    """
    Combine per-instance predicted masks into a single mask via union.

    Instances scoring at or below `score_threshold` are dropped, then each
    surviving instance's own soft mask is binarized independently at
    `instance_mask_threshold` (the standard Mask R-CNN convention, since
    the mask head is trained per-pixel against this decision boundary -
    not a free parameter to sweep) before taking the pixel-wise union
    across instances.

    Previously this combined instances via a confidence-weighted average
    followed by rescaling the result to its own max (see issue #39):
    that's the wrong combination strategy for *disjoint* instances -
    `total_weight` summed the score across every kept instance globally,
    so each vessel's own mask value got diluted by however many other,
    unrelated vessels happened to be in the same image, and the
    per-image max-rescale then made the effective confidence bar float
    depending on each image's score distribution (see #39/#101). Union of
    independently-thresholded instance masks avoids both: each instance
    contributes independently regardless of how many others are present,
    and the output is already binary so no rescaling is needed.

    Args:
        masks (numpy.ndarray): Raw predicted masks, shape (N, 1, H, W)
        scores (numpy.ndarray): Confidence score per instance, shape (N,)
        mask_shape (tuple): (H, W) to use for the empty-mask fallback
        score_threshold (float): Minimum confidence score for an instance to
            be included in the combination
        instance_mask_threshold (float): Threshold used to binarize each
            surviving instance's own soft mask before the union

    Returns:
        numpy.ndarray: Combined binary mask, uint8, values 0 or 255
    """
    keep = scores > score_threshold
    masks = masks[keep]

    if len(masks) > 0:
        instance_binary = masks[:, 0] > instance_mask_threshold
        combined_mask = np.any(instance_binary, axis=0)
        combined_mask = (combined_mask * 255).astype(np.uint8)
    else:
        combined_mask = np.zeros(mask_shape, dtype=np.uint8)

    return combined_mask


def predict_single_image(model, image, device, transform, return_time=False,
                         score_threshold=0.3):
    """
    Perform prediction on a single image.

    Args:
        model (torch.nn.Module): Trained Mask R-CNN model
        image (PIL.Image or str): Input image or path to image
        device (torch.device): Device to run inference on
        transform (callable): Image transformation function
        return_time (bool): Whether to return prediction time
        score_threshold (float): Minimum confidence score for a predicted
            instance to be included in the combined mask

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
        predictions = model([img_tensor])

    end_time = time.time() if return_time else None
    prediction_time = (end_time - start_time) if return_time else None

    # Combine all predicted masks
    pred = predictions[0]
    img_array = np.array(image)
    if len(pred["masks"]) > 0:
        masks = pred["masks"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        combined_mask = combine_prediction_masks(
            masks, scores, img_array.shape[:2], score_threshold)
    else:
        # No predictions - create empty mask
        combined_mask = np.zeros(
            (img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

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
