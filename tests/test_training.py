import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from autoslide import config
from autoslide.pipeline.model.training_utils import (
    setup_directories, create_transforms, initialize_model,
    get_mask_outline, RandomRotation90, generate_negative_samples,
    generate_artificial_vessels
)

# Fixtures for testing


@pytest.fixture
def test_dirs():
    """Create temporary directories for testing"""
    # Use the project's test directory
    test_root = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'tests/test_data')
    os.makedirs(test_root, exist_ok=True)

    # Use the same directory structure as in the config
    plot_dir = os.path.join(test_root, 'plots')
    artifacts_dir = os.path.join(test_root, 'artifacts')

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    yield test_root, plot_dir, artifacts_dir

    # Cleanup after tests
    shutil.rmtree(test_root)


@pytest.fixture
def sample_image_mask():
    """Create a sample image and mask for testing"""
    # Create a simple test image (100x100 RGB)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70, :] = 200  # Create a square in the middle

    # Create a simple mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # Create a smaller square for the mask

    return img, mask


@pytest.fixture
def sample_image_mask_files(test_dirs, sample_image_mask):
    """Save sample image and mask to files for testing"""
    test_root, _, _ = test_dirs

    img_dir = os.path.join(test_root, 'images')
    mask_dir = os.path.join(test_root, 'masks')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img, mask = sample_image_mask

    # Save image and mask
    img_path = os.path.join(img_dir, 'test_image.png')
    mask_path = os.path.join(mask_dir, 'test_image_mask.png')

    plt.imsave(img_path, img)
    plt.imsave(mask_path, mask, cmap='gray')

    return img_dir, mask_dir, ['test_image.png'], ['test_image_mask.png']

# Tests for utility functions


def test_setup_directories(test_dirs):
    """Test that setup_directories creates the necessary directories"""
    test_root, _, _ = test_dirs

    # Test with the test_root as data_dir
    plot_dir, artifacts_dir = setup_directories(test_root)

    assert os.path.exists(plot_dir)
    assert os.path.exists(artifacts_dir)
    assert plot_dir == os.path.join(test_root, 'plots')
    assert artifacts_dir == os.path.join(test_root, 'artifacts')


def test_get_mask_outline(sample_image_mask):
    """Test that get_mask_outline correctly extracts the outline of a mask"""
    _, mask = sample_image_mask

    outline = get_mask_outline(mask)

    # Check that outline is a binary image
    assert np.unique(outline).tolist() == [0, 255]

    # Check that outline has fewer pixels than the original mask
    assert np.sum(outline > 0) < np.sum(mask > 0)


def test_random_rotation90():
    """Test that RandomRotation90 correctly rotates images and masks"""
    # Create a simple test image and mask
    img = Image.new('RGB', (100, 100), color='red')
    # Draw a rectangle in the top-left corner
    for i in range(20, 40):
        for j in range(20, 40):
            img.putpixel((i, j), (0, 255, 0))

    mask = Image.new('L', (100, 100), 0)
    # Draw a rectangle in the top-left corner
    for i in range(20, 40):
        for j in range(20, 40):
            mask.putpixel((i, j), 255)

    # Apply rotation with 100% probability
    rotator = RandomRotation90(p=1.0)
    rotated_img, rotated_mask = rotator(img, mask)

    # Check that the images are different (rotation occurred)
    assert np.array(rotated_img).shape == np.array(img).shape
    assert np.array(rotated_mask).shape == np.array(mask).shape
    assert not np.array_equal(np.array(rotated_img), np.array(img))
    assert not np.array_equal(np.array(rotated_mask), np.array(mask))


def test_create_transforms():
    """Test that create_transforms returns a valid transform"""
    transform = create_transforms()

    # Create a simple test image and mask
    img = Image.new('RGB', (100, 100), color='red')
    mask = Image.new('L', (100, 100), 0)

    # Apply transform
    img_tensor, mask_tensor = transform(img, mask)

    # Check that the output is a tensor with the right shape
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert img_tensor.shape[0] == 3  # RGB channels
    assert mask_tensor.shape[0] == 1  # Single channel mask


def test_generate_negative_samples(sample_image_mask):
    """Test that generate_negative_samples creates valid negative samples"""
    img, mask = sample_image_mask

    neg_img, neg_mask = generate_negative_samples(img, mask)

    # Check that the negative mask is all zeros
    assert np.all(neg_mask == 0)

    # Check that the image has been modified (vessel areas made white)
    assert not np.array_equal(neg_img, img)
    assert np.any(neg_img[mask > 0] == 255)


def test_generate_artificial_vessels(sample_image_mask):
    """Test that generate_artificial_vessels creates valid artificial vessels"""
    img, mask = sample_image_mask

    art_img, art_mask = generate_artificial_vessels(img, mask)

    # Check that the output has the same shape as the input
    assert art_img.shape == img.shape
    assert art_mask.shape == mask.shape

    # Check that the artificial mask is different from the original
    assert not np.array_equal(art_mask, mask)


def test_initialize_model():
    """Test that initialize_model creates a valid Mask R-CNN model"""
    model = initialize_model()

    # Check that the model has the expected structure
    assert hasattr(model, 'roi_heads')
    assert hasattr(model, 'backbone')

    # Check that the model is in training mode by default
    assert model.training

    # Check that the model has the right number of classes (background + vessel)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2

# Integration tests


def test_model_forward_pass():
    """Test that the model can perform a forward pass"""
    model = initialize_model()
    model.eval()  # Set to evaluation mode

    # Create a simple test image
    img = torch.rand(3, 100, 100)

    # Create a simple target
    target = {
        'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.zeros(1, 100, 100, dtype=torch.uint8)
    }
    target['masks'][0, 20:40, 20:40] = 1

    # Test forward pass in training mode
    model.train()
    loss_dict = model([img], [target])

    # Check that the loss dictionary contains the expected keys
    assert 'loss_classifier' in loss_dict
    assert 'loss_box_reg' in loss_dict
    assert 'loss_mask' in loss_dict
    assert 'loss_objectness' in loss_dict
    assert 'loss_rpn_box_reg' in loss_dict

    # Test forward pass in evaluation mode
    model.eval()
    with torch.no_grad():
        predictions = model([img])

    # Check that the predictions have the expected structure
    assert len(predictions) == 1
    assert 'boxes' in predictions[0]
    assert 'labels' in predictions[0]
    assert 'scores' in predictions[0]
    assert 'masks' in predictions[0]
