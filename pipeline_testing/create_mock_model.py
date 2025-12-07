"""
Create a mock Mask R-CNN model for testing purposes.

This script creates a dummy model with random weights that can be used
to test the prediction and fibrosis calculation pipeline without requiring
actual model training.

Usage:
    python pipeline_testing/create_mock_model.py
"""

import os
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def initialize_model():
    """
    Initialize and configure the Mask R-CNN model.
    
    Returns:
        torchvision.models.detection.MaskRCNN: Configured Mask R-CNN model
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    # Configure for binary classification (background + vessel)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    
    return model


def create_mock_model(output_path):
    """
    Create a mock model with random weights for testing.
    
    Args:
        output_path: Path where the model should be saved
    """
    print("Creating mock Mask R-CNN model...")
    print(f"Output path: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize model with random weights
    model = initialize_model()
    
    # Save the model
    torch.save(model.state_dict(), output_path)
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"✓ Mock model created successfully")
    print(f"✓ Model size: {file_size:.2f} MB")
    print(f"✓ Saved to: {output_path}")
    print("\nNote: This is a mock model with random weights for testing purposes only.")
    print("It will produce random predictions and should not be used for actual analysis.")


def main():
    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Default output path
    output_path = os.path.join(
        project_root, 
        'autoslide', 
        'src', 
        'pipeline', 
        'model', 
        'artifacts',
        'best_val_mask_rcnn_model.pth'
    )
    
    # Allow custom output path from command line
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    create_mock_model(output_path)


if __name__ == '__main__':
    main()
