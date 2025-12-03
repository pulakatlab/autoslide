"""
Model evaluation script for assessing prediction accuracy and speed.

This module provides:
- Intersection over Union (IoU) calculation for segmentation accuracy
- Prediction speed benchmarking
- Comprehensive evaluation metrics and visualizations
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2 as cv
from scipy.optimize import curve_fit
import pickle
import hashlib
import json

# Import config and utilities
from src import config
from src.pipeline.model.data_preprocessing import load_data, split_train_val
from src.pipeline.model.prediction_utils import load_model, predict_single_image, setup_device

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
plot_dir = config['plot_dirs']


#############################################################################
# IoU and Accuracy Metrics
#############################################################################

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary masks.

    Args:
        pred_mask (numpy.ndarray): Predicted mask (0-1 or 0-255)
        true_mask (numpy.ndarray): Ground truth mask (0-1 or 0-255)
        threshold (float): Threshold for binarizing predicted mask

    Returns:
        float: IoU score between 0 and 1
    """
    try:
        # Ensure masks are binary
        if pred_mask.max() > 1:
            pred_mask = pred_mask / 255.0
        if true_mask.max() > 1:
            true_mask = true_mask / 255.0

        # Binarize predicted mask
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        true_binary = (true_mask > threshold).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, true_binary).sum()
        union = np.logical_or(pred_binary, true_binary).sum()

        # Handle edge case where both masks are empty
        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0


def calculate_dice_coefficient(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Dice coefficient for binary masks.

    Args:
        pred_mask (numpy.ndarray): Predicted mask (0-1 or 0-255)
        true_mask (numpy.ndarray): Ground truth mask (0-1 or 0-255)
        threshold (float): Threshold for binarizing predicted mask

    Returns:
        float: Dice coefficient between 0 and 1
    """
    # Ensure masks are binary
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if true_mask.max() > 1:
        true_mask = true_mask / 255.0

    # Binarize predicted mask
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)

    # Calculate Dice coefficient
    intersection = np.logical_and(pred_binary, true_binary).sum()
    total = pred_binary.sum() + true_binary.sum()

    # Handle edge case where both masks are empty
    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2 * intersection / total


def calculate_pixel_accuracy(pred_mask, true_mask, threshold=0.5):
    """
    Calculate pixel-wise accuracy.

    Args:
        pred_mask (numpy.ndarray): Predicted mask (0-1 or 0-255)
        true_mask (numpy.ndarray): Ground truth mask (0-1 or 0-255)
        threshold (float): Threshold for binarizing predicted mask

    Returns:
        float: Pixel accuracy between 0 and 1
    """
    # Ensure masks are binary
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if true_mask.max() > 1:
        true_mask = true_mask / 255.0

    # Binarize predicted mask
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)

    # Calculate pixel accuracy
    correct_pixels = (pred_binary == true_binary).sum()
    total_pixels = pred_binary.size

    return correct_pixels / total_pixels


def calculate_confidence_metrics(pred_mask, true_mask):
    """
    Calculate prediction confidence inside and outside labeled bounds.

    Args:
        pred_mask (numpy.ndarray): Predicted mask (0-1 or 0-255)
        true_mask (numpy.ndarray): Ground truth mask (0-1 or 0-255)

    Returns:
        dict: Dictionary containing confidence metrics
    """
    # Normalize prediction mask to 0-1 range
    if pred_mask.max() > 1:
        pred_mask_norm = pred_mask / 255.0
    else:
        pred_mask_norm = pred_mask.copy()

    # Normalize true mask to 0-1 range
    if true_mask.max() > 1:
        true_mask_norm = true_mask / 255.0
    else:
        true_mask_norm = true_mask.copy()

    # Create binary masks for labeled regions
    vessel_regions = (true_mask_norm > 0.5).astype(bool)
    background_regions = (true_mask_norm <= 0.5).astype(bool)

    # Calculate confidence metrics
    results = {}

    # Confidence inside vessel regions (should be high)
    if np.any(vessel_regions):
        vessel_confidences = pred_mask_norm[vessel_regions]
        results['mean_confidence_in_vessels'] = np.mean(vessel_confidences)
        results['std_confidence_in_vessels'] = np.std(vessel_confidences)
        results['median_confidence_in_vessels'] = np.median(vessel_confidences)
        results['num_vessel_pixels'] = len(vessel_confidences)
    else:
        results['mean_confidence_in_vessels'] = 0.0
        results['std_confidence_in_vessels'] = 0.0
        results['median_confidence_in_vessels'] = 0.0
        results['num_vessel_pixels'] = 0

    # Confidence outside vessel regions (should be low)
    if np.any(background_regions):
        background_confidences = pred_mask_norm[background_regions]
        results['mean_confidence_in_background'] = np.mean(
            background_confidences)
        results['std_confidence_in_background'] = np.std(
            background_confidences)
        results['median_confidence_in_background'] = np.median(
            background_confidences)
        results['num_background_pixels'] = len(background_confidences)
    else:
        results['mean_confidence_in_background'] = 0.0
        results['std_confidence_in_background'] = 0.0
        results['median_confidence_in_background'] = 0.0
        results['num_background_pixels'] = 0

    # Calculate confidence separation (higher is better)
    if results['num_vessel_pixels'] > 0 and results['num_background_pixels'] > 0:
        results['confidence_separation'] = (
            results['mean_confidence_in_vessels'] -
            results['mean_confidence_in_background']
        )
    else:
        results['confidence_separation'] = 0.0

    return results


#############################################################################
# Caching Functions
#############################################################################

def get_cache_key(model_path, img_name, transform_params=None):
    """
    Generate a unique cache key for a prediction based on model and image.
    
    Args:
        model_path (str): Path to the model file
        img_name (str): Image filename
        transform_params (dict): Transform parameters (optional)
    
    Returns:
        str: Unique cache key
    """
    try:
        # Get model file hash
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_hash = hashlib.md5(f.read()).hexdigest()[:8]
        else:
            print(f"Warning: Model file not found at {model_path}, using 'unknown' hash")
            model_hash = "unknown"
        
        # Create cache key
        key_data = f"{model_hash}_{img_name}"
        if transform_params:
            key_data += f"_{str(transform_params)}"
        
        cache_key = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return cache_key
    except Exception as e:
        print(f"Error generating cache key for {img_name}: {e}")
        return hashlib.md5(f"fallback_{img_name}".encode()).hexdigest()[:16]


def save_prediction_cache(cache_dir, cache_key, pred_mask, pred_time, metrics=None):
    """
    Save prediction results to cache.
    
    Args:
        cache_dir (str): Cache directory
        cache_key (str): Unique cache key
        pred_mask (numpy.ndarray): Predicted mask
        pred_time (float): Prediction time
        metrics (dict): Optional metrics to cache
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_data = {
            'pred_mask': pred_mask,
            'pred_time': pred_time,
            'metrics': metrics or {}
        }
        
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Could not save prediction to cache (key: {cache_key}): {e}")


def load_prediction_cache(cache_dir, cache_key):
    """
    Load prediction results from cache.
    
    Args:
        cache_dir (str): Cache directory
        cache_key (str): Unique cache key
    
    Returns:
        tuple: (pred_mask, pred_time, metrics) or (None, None, None) if not found
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_path):
        return None, None, None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        return cache_data['pred_mask'], cache_data['pred_time'], cache_data.get('metrics', {})
    except Exception as e:
        print(f"Warning: Could not load cache for key {cache_key}: {e}")
        return None, None, None


def clear_prediction_cache(cache_dir):
    """
    Clear all cached predictions.
    
    Args:
        cache_dir (str): Cache directory to clear
    """
    try:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Successfully cleared prediction cache: {cache_dir}")
        else:
            print(f"Cache directory does not exist: {cache_dir}")
    except Exception as e:
        print(f"Error clearing prediction cache: {e}")


def get_cache_info(cache_dir):
    """
    Get information about the prediction cache.
    
    Args:
        cache_dir (str): Cache directory
    
    Returns:
        dict: Cache information
    """
    if not os.path.exists(cache_dir):
        return {'num_files': 0, 'total_size_mb': 0}
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)
    
    return {
        'num_files': len(cache_files),
        'total_size_mb': total_size / (1024 * 1024)
    }


#############################################################################
# Model Prediction Functions
#############################################################################


#############################################################################
# Evaluation Pipeline
#############################################################################

def evaluate_model_accuracy_and_speed(model, val_imgs, val_masks, img_dir, mask_dir,
                                      aug_img_dir, aug_mask_dir, device, transform,
                                      output_dir,
                                      max_samples=None,
                                      num_warmup=10,
                                      use_cache=True,
                                      model_path=None,
                                      ):
    """
    Evaluate model accuracy and speed on validation set.

    Args:
        model (torch.nn.Module): Trained model
        val_imgs (list): Validation image filenames
        val_masks (list): Validation mask filenames
        img_dir (str): Original images directory
        mask_dir (str): Original masks directory
        aug_img_dir (str): Augmented images directory
        aug_mask_dir (str): Augmented masks directory
        device (torch.device): Device for inference
        transform (callable): Image transformation
        output_dir (str): Directory to save prediction visualizations
        max_samples (int): Maximum number of samples to evaluate
        num_warmup (int): Number of warmup iterations before timing
        use_cache (bool): Whether to use prediction caching
        model_path (str): Path to model file (for cache key generation)

    Returns:
        tuple: (accuracy_results, speed_results) dictionaries
    """
    print(
        f'Evaluating model accuracy and speed on {len(val_imgs)} validation samples...')

    model.eval()
    
    # Setup cache directory
    cache_dir = None
    if use_cache:
        cache_dir = os.path.join(artifacts_dir, 'prediction_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_info = get_cache_info(cache_dir)
        print(f'Using prediction cache: {cache_info["num_files"]} files, {cache_info["total_size_mb"]:.1f} MB')

    # Limit samples if specified
    if max_samples and max_samples < len(val_imgs):
        indices = np.random.choice(len(val_imgs), max_samples, replace=False)
        eval_imgs = [val_imgs[i] for i in indices]
        eval_masks = [val_masks[i] for i in indices]
        print(f'Limited evaluation to {max_samples} random samples')
    else:
        eval_imgs = val_imgs
        eval_masks = val_masks

    # Warmup phase for accurate timing
    if num_warmup > 0:
        print(f'Warming up with {num_warmup} iterations...')
        warmup_imgs = np.random.choice(eval_imgs, min(num_warmup, len(eval_imgs)), replace=True)
        for img_name in tqdm(warmup_imgs, desc='Warmup'):
            try:
                if 'aug_' in img_name:
                    image = Image.open(os.path.join(aug_img_dir, img_name)).convert("RGB")
                else:
                    image = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
                _ = predict_single_image(model, image, device, transform, return_time=False)
            except Exception as e:
                print(f'Error during warmup with {img_name}: {e}')
                continue

    # Metrics storage
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    prediction_times = []
    positive_areas = []  # Store positive area in ground truth masks

    # Confidence metrics storage
    confidence_in_vessels = []
    confidence_in_background = []
    confidence_separations = []
    per_image_confidence_metrics = []

    # Evaluate each sample (accuracy and timing together)
    cache_hits = 0
    cache_misses = 0
    processed_count = 0
    error_count = 0
    
    print(f"Starting evaluation of {len(eval_imgs)} samples...")
    
    for img_name, mask_name in tqdm(zip(eval_imgs, eval_masks),
                                    total=len(eval_imgs),
                                    desc='Evaluating accuracy and speed'):
        try:
            # Load image and mask
            if 'aug_' in img_name:
                image_path = os.path.join(aug_img_dir, img_name)
                mask_path = os.path.join(aug_mask_dir, mask_name)
            else:
                image_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, mask_name)
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                error_count += 1
                continue
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found: {mask_path}")
                error_count += 1
                continue
            
            image = Image.open(image_path).convert("RGB")
            true_mask = np.array(Image.open(mask_path))

            # Flatten labels in mask
            true_mask = (true_mask > 0).astype(
                np.uint8) * 255  # Ensure binary mask

            # Calculate positive area (fraction of pixels that are positive)
            positive_area = np.sum(true_mask > 0) / true_mask.size

            # Try to load from cache first
            pred_mask = None
            pred_time = None
            cached_metrics = {}
            
            if use_cache and cache_dir and model_path:
                cache_key = get_cache_key(model_path, img_name)
                pred_mask, pred_time, cached_metrics = load_prediction_cache(cache_dir, cache_key)
                
                if pred_mask is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1

            # Get prediction if not cached
            if pred_mask is None:
                pred_time, pred_mask = predict_single_image(
                    model, image, device, transform, return_time=True)
                
                # Save to cache if enabled
                if use_cache and cache_dir and model_path:
                    cache_key = get_cache_key(model_path, img_name)
                    save_prediction_cache(cache_dir, cache_key, pred_mask, pred_time)

            # Calculate metrics
            iou = calculate_iou(pred_mask, true_mask)
            dice = calculate_dice_coefficient(pred_mask, true_mask)
            pixel_acc = calculate_pixel_accuracy(pred_mask, true_mask)

            # Calculate confidence metrics
            confidence_metrics = calculate_confidence_metrics(
                pred_mask, true_mask)

            # Store results
            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            prediction_times.append(pred_time)
            positive_areas.append(positive_area)

            # Store confidence metrics
            confidence_in_vessels.append(
                confidence_metrics['mean_confidence_in_vessels'])
            confidence_in_background.append(
                confidence_metrics['mean_confidence_in_background'])
            confidence_separations.append(
                confidence_metrics['confidence_separation'])

            # Store per-image confidence metrics with image name
            per_image_metrics = confidence_metrics.copy()
            per_image_metrics['image_name'] = img_name
            per_image_metrics['iou'] = iou
            per_image_metrics['dice'] = dice
            per_image_confidence_metrics.append(per_image_metrics)

            # Plot sample predictions
            try:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image)
                axes[0].set_title(f'Original Image\n{img_name}')
                axes[0].axis('off')
                axes[1].imshow(true_mask, cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title(
                    f'Predicted Mask\nIoU: {iou:.3f}, Dice: {dice:.3f}')
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'pred_{img_name}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as plot_error:
                print(f"Warning: Could not save visualization for {img_name}: {plot_error}")

            processed_count += 1

        except Exception as e:
            print(f'Error processing {img_name}: {e}')
            error_count += 1
            continue
    
    print(f"Evaluation completed: {processed_count} samples processed successfully, {error_count} errors")

    # Print cache statistics
    if use_cache:
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            cache_hit_rate = cache_hits / total_requests * 100
            print(f'Cache statistics: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.1f}% hit rate)')

    # Calculate accuracy summary statistics
    accuracy_results = {
        'num_samples': len(iou_scores),
        'mean_iou': np.mean(iou_scores),
        'std_iou': np.std(iou_scores),
        'median_iou': np.median(iou_scores),
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'median_dice': np.median(dice_scores),
        'mean_pixel_accuracy': np.mean(pixel_accuracies),
        'std_pixel_accuracy': np.std(pixel_accuracies),
        'median_pixel_accuracy': np.median(pixel_accuracies),
        'mean_prediction_time': np.mean(prediction_times),
        'std_prediction_time': np.std(prediction_times),
        'median_prediction_time': np.median(prediction_times),
        'fps': 1.0 / np.mean(prediction_times),
        'iou_scores': iou_scores,
        'dice_scores': dice_scores,
        'pixel_accuracies': pixel_accuracies,
        'prediction_times': prediction_times,
        'positive_areas': positive_areas,

        # Confidence metrics
        'mean_confidence_in_vessels': np.mean(confidence_in_vessels),
        'std_confidence_in_vessels': np.std(confidence_in_vessels),
        'median_confidence_in_vessels': np.median(confidence_in_vessels),
        'mean_confidence_in_background': np.mean(confidence_in_background),
        'std_confidence_in_background': np.std(confidence_in_background),
        'median_confidence_in_background': np.median(confidence_in_background),
        'mean_confidence_separation': np.mean(confidence_separations),
        'std_confidence_separation': np.std(confidence_separations),
        'median_confidence_separation': np.median(confidence_separations),
        'confidence_in_vessels': confidence_in_vessels,
        'confidence_in_background': confidence_in_background,
        'confidence_separations': confidence_separations,
        'per_image_confidence_metrics': per_image_confidence_metrics
    }

    # Calculate speed summary statistics
    speed_results = {
        'num_samples': len(prediction_times),
        'mean_time': np.mean(prediction_times),
        'std_time': np.std(prediction_times),
        'median_time': np.median(prediction_times),
        'min_time': np.min(prediction_times),
        'max_time': np.max(prediction_times),
        'mean_fps': 1.0 / np.mean(prediction_times),
        'median_fps': 1.0 / np.median(prediction_times),
        'times': prediction_times
    }

    return accuracy_results, speed_results


def benchmark_prediction_speed(model, val_imgs, img_dir, aug_img_dir,
                               device, transform, num_warmup=10, num_benchmark=100,
                               use_cache=True, model_path=None):
    """
    Benchmark prediction speed on a subset of validation images.

    Args:
        model (torch.nn.Module): Trained model
        val_imgs (list): Validation image filenames
        img_dir (str): Original images directory
        aug_img_dir (str): Augmented images directory
        device (torch.device): Device for inference
        transform (callable): Image transformation
        num_warmup (int): Number of warmup iterations
        num_benchmark (int): Number of benchmark iterations
        use_cache (bool): Whether to use prediction caching
        model_path (str): Path to model file (for cache key generation)

    Returns:
        dict: Speed benchmark results
    """
    print(f'Benchmarking prediction speed...')

    model.eval()
    
    # Setup cache directory
    cache_dir = None
    if use_cache:
        cache_dir = os.path.join(artifacts_dir, 'prediction_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_info = get_cache_info(cache_dir)
        print(f'Using prediction cache for speed benchmark: {cache_info["num_files"]} files, {cache_info["total_size_mb"]:.1f} MB')

    # Select random images for benchmarking
    benchmark_imgs = np.random.choice(val_imgs,
                                      min(num_warmup + num_benchmark,
                                          len(val_imgs)),
                                      replace=False)

    # Warmup phase
    print(f'Warming up with {num_warmup} iterations...')
    warmup_imgs = np.random.choice(benchmark_imgs, num_warmup, replace=True)
    for img_name in tqdm(warmup_imgs, desc='Warmup'):
        try:
            if 'aug_' in img_name:
                image_path = os.path.join(aug_img_dir, img_name)
            else:
                image_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(image_path):
                continue
                
            image = Image.open(image_path).convert("RGB")
            _ = predict_single_image(model, image, device, transform, return_time=False)
        except Exception:
            continue

    # Benchmark phase
    print(f'Benchmarking with {num_benchmark} iterations...')
    benchmark_times = []
    benchmark_errors = 0
    cache_hits = 0
    cache_misses = 0

    for img_name in tqdm(benchmark_imgs, desc='Benchmarking speed'):
        try:
            if 'aug_' in img_name:
                image_path = os.path.join(aug_img_dir, img_name)
            else:
                image_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Benchmark image not found: {image_path}")
                benchmark_errors += 1
                continue
                
            image = Image.open(image_path).convert("RGB")

            # Try to load from cache first
            pred_time = None
            pred_mask = None
            if use_cache and cache_dir and model_path:
                cache_key = get_cache_key(model_path, img_name)
                pred_mask, cached_time, _ = load_prediction_cache(cache_dir, cache_key)
                if pred_mask is not None and cached_time is not None:
                    pred_time = cached_time
                    cache_hits += 1
                else:
                    cache_misses += 1

            # Get fresh prediction if not cached
            if pred_time is None:
                pred_time, pred_mask = predict_single_image(
                    model, image, device, transform, return_time=True)
                
                # Save to cache if enabled
                if use_cache and cache_dir and model_path:
                    cache_key = get_cache_key(model_path, img_name)
                    save_prediction_cache(cache_dir, cache_key, pred_mask, pred_time)
            
            benchmark_times.append(pred_time)
            
        except Exception as e:
            print(f"Error during benchmark with {img_name}: {e}")
            benchmark_errors += 1
            continue
    
    if benchmark_errors > 0:
        print(f"Benchmark completed with {benchmark_errors} errors out of {len(benchmark_imgs)} samples")
    
    # Print cache statistics
    if use_cache:
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            cache_hit_rate = cache_hits / total_requests * 100
            print(f'Speed benchmark cache statistics: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.1f}% hit rate)')

    # Calculate speed metrics
    speed_results = {
        'num_samples': len(benchmark_times),
        'mean_time': np.mean(benchmark_times),
        'std_time': np.std(benchmark_times),
        'median_time': np.median(benchmark_times),
        'min_time': np.min(benchmark_times),
        'max_time': np.max(benchmark_times),
        'mean_fps': 1.0 / np.mean(benchmark_times),
        'median_fps': 1.0 / np.median(benchmark_times),
        'times': benchmark_times
    }

    return speed_results


#############################################################################
# Visualization Functions
#############################################################################

def logistic_function(x, L, k, x0, b):
    """
    Logistic function: L / (1 + exp(-k*(x-x0))) + b
    
    Args:
        x: input values
        L: maximum value of the curve
        k: steepness of the curve
        x0: x-value of the sigmoid's midpoint (inflection point)
        b: y-offset
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_logistic_curve(x_data, y_data):
    """
    Fit a logistic curve to the data and return parameters and fitted curve.
    
    Args:
        x_data: x values
        y_data: y values
        
    Returns:
        tuple: (fitted_params, x_fit, y_fit, inflection_point)
    """
    try:
        # Initial parameter guesses
        L_guess = np.max(y_data) - np.min(y_data)  # amplitude
        b_guess = np.min(y_data)  # offset
        x0_guess = np.median(x_data)  # inflection point guess
        k_guess = 1.0  # steepness
        
        # Fit the logistic curve
        popt, _ = curve_fit(logistic_function, x_data, y_data, 
                           p0=[L_guess, k_guess, x0_guess, b_guess],
                           maxfev=5000)
        
        # Generate fitted curve
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_fit = logistic_function(x_fit, *popt)
        
        # Inflection point is at x0
        inflection_point = popt[2]
        inflection_y = logistic_function(inflection_point, *popt)
        
        return popt, x_fit, y_fit, (inflection_point, inflection_y)
    
    except Exception as e:
        print(f"Warning: Could not fit logistic curve: {e}")
        return None, None, None, None


def plot_metrics_vs_positive_area(accuracy_results, plot_dir):
    """
    Plot IoU and Dice scores vs positive area in ground truth masks with logistic curve fitting.
    Also creates separate plots for IoU based on positive area fraction thresholds.

    Args:
        accuracy_results (dict): Results from accuracy evaluation
        plot_dir (str): Directory to save plots
    """
    print("Creating metrics vs positive area plots...")
    
    try:
        eval_plot_dir = os.path.join(plot_dir, 'model_evaluation')
        os.makedirs(eval_plot_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating evaluation plot directory: {e}")
        return

    # Convert to numpy arrays for easier manipulation
    positive_areas = np.array(accuracy_results['positive_areas'])
    iou_scores = np.array(accuracy_results['iou_scores'])
    dice_scores = np.array(accuracy_results['dice_scores'])

    # Create main plot with IoU and Dice vs positive area
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # IoU vs Positive Area
    axes[0].scatter(positive_areas, iou_scores, alpha=0.6, s=30)
    axes[0].set_xlabel('Positive Area (fraction of pixels)')
    axes[0].set_ylabel('IoU Score')
    axes[0].set_title('IoU Score vs Positive Area in Ground Truth')
    axes[0].grid(True, alpha=0.3)
    
    # Fit logistic curve for IoU
    iou_params, x_fit_iou, y_fit_iou, iou_inflection = fit_logistic_curve(
        positive_areas, iou_scores
    )
    
    if iou_params is not None:
        axes[0].plot(x_fit_iou, y_fit_iou, "r-", alpha=0.8, linewidth=2,
                    label=f'Logistic fit (k={iou_params[1]:.2f})')
        # Mark inflection point
        axes[0].plot(iou_inflection[0], iou_inflection[1], 'ro', markersize=8,
                    label=f'Inflection point ({iou_inflection[0]:.3f}, {iou_inflection[1]:.3f})')
    else:
        # Fallback to linear fit if logistic fails
        z_iou = np.polyfit(positive_areas, iou_scores, 1)
        p_iou = np.poly1d(z_iou)
        x_trend = np.linspace(min(positive_areas), max(positive_areas), 100)
        axes[0].plot(x_trend, p_iou(x_trend), "r--", alpha=0.8, 
                    label=f'Linear fit: slope={z_iou[0]:.3f}')
    
    axes[0].legend()

    # Dice vs Positive Area
    axes[1].scatter(positive_areas, dice_scores, alpha=0.6, s=30, color='orange')
    axes[1].set_xlabel('Positive Area (fraction of pixels)')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Dice Score vs Positive Area in Ground Truth')
    axes[1].grid(True, alpha=0.3)
    
    # Fit logistic curve for Dice
    dice_params, x_fit_dice, y_fit_dice, dice_inflection = fit_logistic_curve(
        positive_areas, dice_scores
    )
    
    if dice_params is not None:
        axes[1].plot(x_fit_dice, y_fit_dice, "r-", alpha=0.8, linewidth=2,
                    label=f'Logistic fit (k={dice_params[1]:.2f})')
        # Mark inflection point
        axes[1].plot(dice_inflection[0], dice_inflection[1], 'ro', markersize=8,
                    label=f'Inflection point ({dice_inflection[0]:.3f}, {dice_inflection[1]:.3f})')
    else:
        # Fallback to linear fit if logistic fails
        z_dice = np.polyfit(positive_areas, dice_scores, 1)
        p_dice = np.poly1d(z_dice)
        x_trend = np.linspace(min(positive_areas), max(positive_areas), 100)
        axes[1].plot(x_trend, p_dice(x_trend), "r--", alpha=0.8,
                    label=f'Linear fit: slope={z_dice[0]:.3f}')
    
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(eval_plot_dir, 'metrics_vs_positive_area.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate IoU plots for positive area fraction < 0.15 and >= 0.15
    threshold = 0.015
    low_area_mask = positive_areas < threshold
    high_area_mask = positive_areas >= threshold
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # IoU for positive area < 0.15
    if np.any(low_area_mask):
        low_areas = positive_areas[low_area_mask]
        low_ious = iou_scores[low_area_mask]
        
        axes[0].scatter(low_areas, low_ious, alpha=0.6, s=30, color='blue')
        axes[0].set_xlabel('Positive Area (fraction of pixels)')
        axes[0].set_ylabel('IoU Score')
        axes[0].set_title(f'IoU Score vs Positive Area < {threshold}\n(n={len(low_ious)} samples)')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_iou_low = np.mean(low_ious)
        std_iou_low = np.std(low_ious)
        axes[0].axhline(mean_iou_low, color='red', linestyle='--', alpha=0.8,
                       label=f'Mean: {mean_iou_low:.3f} ± {std_iou_low:.3f}')
        axes[0].legend()
        
        # Fit trend line if enough points
        if len(low_areas) > 2:
            z_low = np.polyfit(low_areas, low_ious, 1)
            p_low = np.poly1d(z_low)
            x_trend_low = np.linspace(min(low_areas), max(low_areas), 100)
            axes[0].plot(x_trend_low, p_low(x_trend_low), "g--", alpha=0.8,
                        label=f'Trend: slope={z_low[0]:.3f}')
            axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, f'No samples with positive area < {threshold}', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f'IoU Score vs Positive Area < {threshold}')
    
    # IoU for positive area >= 0.15
    if np.any(high_area_mask):
        high_areas = positive_areas[high_area_mask]
        high_ious = iou_scores[high_area_mask]
        
        axes[1].scatter(high_areas, high_ious, alpha=0.6, s=30, color='green')
        axes[1].set_xlabel('Positive Area (fraction of pixels)')
        axes[1].set_ylabel('IoU Score')
        axes[1].set_title(f'IoU Score vs Positive Area ≥ {threshold}\n(n={len(high_ious)} samples)')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_iou_high = np.mean(high_ious)
        std_iou_high = np.std(high_ious)
        axes[1].axhline(mean_iou_high, color='red', linestyle='--', alpha=0.8,
                       label=f'Mean: {mean_iou_high:.3f} ± {std_iou_high:.3f}')
        axes[1].legend()
        
        # Fit trend line if enough points
        if len(high_areas) > 2:
            z_high = np.polyfit(high_areas, high_ious, 1)
            p_high = np.poly1d(z_high)
            x_trend_high = np.linspace(min(high_areas), max(high_areas), 100)
            axes[1].plot(x_trend_high, p_high(x_trend_high), "g--", alpha=0.8,
                        label=f'Trend: slope={z_high[0]:.3f}')
            axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, f'No samples with positive area ≥ {threshold}', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'IoU Score vs Positive Area ≥ {threshold}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_plot_dir, 'iou_by_positive_area_threshold.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Metrics vs positive area plot saved to {eval_plot_dir}/metrics_vs_positive_area.png')
    print(f'IoU by positive area threshold plot saved to {eval_plot_dir}/iou_by_positive_area_threshold.png')

    # Print correlation statistics and logistic fit results
    iou_corr = np.corrcoef(positive_areas, iou_scores)[0, 1]
    dice_corr = np.corrcoef(positive_areas, dice_scores)[0, 1]
    
    print(f'Correlation between positive area and IoU: {iou_corr:.3f}')
    print(f'Correlation between positive area and Dice: {dice_corr:.3f}')
    
    # Print statistics for different positive area ranges
    if np.any(low_area_mask):
        low_ious = iou_scores[low_area_mask]
        print(f'IoU for positive area < {threshold}: {np.mean(low_ious):.3f} ± {np.std(low_ious):.3f} (n={len(low_ious)})')
    
    if np.any(high_area_mask):
        high_ious = iou_scores[high_area_mask]
        print(f'IoU for positive area ≥ {threshold}: {np.mean(high_ious):.3f} ± {np.std(high_ious):.3f} (n={len(high_ious)})')
    
    if iou_params is not None:
        print(f'IoU logistic fit - Inflection point: {iou_inflection[0]:.3f}, Steepness: {iou_params[1]:.3f}')
    if dice_params is not None:
        print(f'Dice logistic fit - Inflection point: {dice_inflection[0]:.3f}, Steepness: {dice_params[1]:.3f}')


def plot_evaluation_results(accuracy_results, speed_results, plot_dir):
    """
    Create comprehensive evaluation plots.

    Args:
        accuracy_results (dict): Results from accuracy evaluation
        speed_results (dict): Results from speed benchmarking
        plot_dir (str): Directory to save plots
    """
    print("Creating comprehensive evaluation plots...")
    
    try:
        eval_plot_dir = os.path.join(plot_dir, 'model_evaluation')
        os.makedirs(eval_plot_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating evaluation plot directory: {e}")
        return

    # Plot 1: Accuracy metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # IoU distribution
    axes[0, 0].hist(accuracy_results['iou_scores'],
                    bins=30, alpha=0.7, color='blue')
    axes[0, 0].axvline(accuracy_results['mean_iou'], color='red', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_iou"]:.3f}')
    axes[0, 0].axvline(accuracy_results['median_iou'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_iou"]:.3f}')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('IoU Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dice coefficient distribution
    axes[0, 1].hist(accuracy_results['dice_scores'],
                    bins=30, alpha=0.7, color='orange')
    axes[0, 1].axvline(accuracy_results['mean_dice'], color='red', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_dice"]:.3f}')
    axes[0, 1].axvline(accuracy_results['median_dice'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_dice"]:.3f}')
    axes[0, 1].set_xlabel('Dice Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Dice Coefficient Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Pixel accuracy distribution
    axes[1, 0].hist(accuracy_results['pixel_accuracies'],
                    bins=30, alpha=0.7, color='green')
    axes[1, 0].axvline(accuracy_results['mean_pixel_accuracy'], color='red', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_pixel_accuracy"]:.3f}')
    axes[1, 0].axvline(accuracy_results['median_pixel_accuracy'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_pixel_accuracy"]:.3f}')
    axes[1, 0].set_xlabel('Pixel Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Pixel Accuracy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prediction time distribution
    axes[1, 1].hist(accuracy_results['prediction_times'],
                    bins=30, alpha=0.7, color='purple')
    axes[1, 1].axvline(accuracy_results['mean_prediction_time'], color='red', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_prediction_time"]:.3f}s')
    axes[1, 1].axvline(accuracy_results['median_prediction_time'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_prediction_time"]:.3f}s')
    axes[1, 1].set_xlabel('Prediction Time (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Time Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_plot_dir, 'accuracy_metrics_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Confidence metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confidence in vessel regions
    axes[0, 0].hist(accuracy_results['confidence_in_vessels'],
                    bins=30, alpha=0.7, color='red')
    axes[0, 0].axvline(accuracy_results['mean_confidence_in_vessels'], color='blue', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_confidence_in_vessels"]:.3f}')
    axes[0, 0].axvline(accuracy_results['median_confidence_in_vessels'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_confidence_in_vessels"]:.3f}')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Confidence in Vessel Regions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Confidence in background regions
    axes[0, 1].hist(accuracy_results['confidence_in_background'],
                    bins=30, alpha=0.7, color='blue')
    axes[0, 1].axvline(accuracy_results['mean_confidence_in_background'], color='red', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_confidence_in_background"]:.3f}')
    axes[0, 1].axvline(accuracy_results['median_confidence_in_background'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_confidence_in_background"]:.3f}')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence in Background Regions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confidence separation
    axes[1, 0].hist(accuracy_results['confidence_separations'],
                    bins=30, alpha=0.7, color='purple')
    axes[1, 0].axvline(accuracy_results['mean_confidence_separation'], color='red', linestyle='--',
                       label=f'Mean: {accuracy_results["mean_confidence_separation"]:.3f}')
    axes[1, 0].axvline(accuracy_results['median_confidence_separation'], color='green', linestyle='--',
                       label=f'Median: {accuracy_results["median_confidence_separation"]:.3f}')
    axes[1, 0].set_xlabel('Confidence Separation')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Confidence Separation (Vessel - Background)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Scatter plot: Confidence separation vs IoU
    axes[1, 1].scatter(accuracy_results['confidence_separations'],
                       accuracy_results['iou_scores'], alpha=0.6)
    axes[1, 1].set_xlabel('Confidence Separation')
    axes[1, 1].set_ylabel('IoU Score')
    axes[1, 1].set_title('Confidence Separation vs IoU')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_plot_dir, 'confidence_metrics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Speed benchmark results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Speed distribution
    axes[0].hist(speed_results['times'], bins=30, alpha=0.7, color='red')
    axes[0].axvline(speed_results['mean_time'], color='blue', linestyle='--',
                    label=f'Mean: {speed_results["mean_time"]:.4f}s')
    axes[0].axvline(speed_results['median_time'], color='green', linestyle='--',
                    label=f'Median: {speed_results["median_time"]:.4f}s')
    axes[0].set_xlabel('Prediction Time (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Speed Benchmark Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FPS visualization
    fps_values = [1.0 / t for t in speed_results['times']]
    axes[1].hist(fps_values, bins=30, alpha=0.7, color='cyan')
    axes[1].axvline(speed_results['mean_fps'], color='blue', linestyle='--',
                    label=f'Mean: {speed_results["mean_fps"]:.2f} FPS')
    axes[1].axvline(speed_results['median_fps'], color='green', linestyle='--',
                    label=f'Median: {speed_results["median_fps"]:.2f} FPS')
    axes[1].set_xlabel('Frames Per Second (FPS)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('FPS Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(eval_plot_dir, 'speed_benchmark.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Evaluation plots saved to {eval_plot_dir}')


def create_sample_predictions_plot(model, val_imgs, val_masks, img_dir, mask_dir,
                                   aug_img_dir, aug_mask_dir, device, transform,
                                   plot_dir, num_samples=6, use_cache=True, model_path=None):
    """
    Create a plot showing sample predictions vs ground truth.

    Args:
        model (torch.nn.Module): Trained model
        val_imgs (list): Validation image filenames
        val_masks (list): Validation mask filenames
        img_dir (str): Original images directory
        mask_dir (str): Original masks directory
        aug_img_dir (str): Augmented images directory
        aug_mask_dir (str): Augmented masks directory
        device (torch.device): Device for inference
        transform (callable): Image transformation
        plot_dir (str): Directory to save plots
        num_samples (int): Number of sample predictions to show
        use_cache (bool): Whether to use prediction caching
        model_path (str): Path to model file (for cache key generation)
    """
    print(f"Creating sample predictions plot with {num_samples} samples...")
    
    try:
        eval_plot_dir = os.path.join(plot_dir, 'model_evaluation')
        os.makedirs(eval_plot_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating evaluation plot directory: {e}")
        return

    model.eval()
    
    # Setup cache directory
    cache_dir = None
    if use_cache:
        cache_dir = os.path.join(artifacts_dir, 'prediction_cache')
        os.makedirs(cache_dir, exist_ok=True)

    # Select random samples
    sample_indices = np.random.choice(
        len(val_imgs), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    sample_errors = 0
    
    for i, idx in enumerate(sample_indices):
        try:
            img_name = val_imgs[idx]
            mask_name = val_masks[idx]

            # Load image and mask
            if 'aug_' in img_name:
                image_path = os.path.join(aug_img_dir, img_name)
                mask_path = os.path.join(aug_mask_dir, mask_name)
            else:
                image_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, mask_name)
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"Warning: Sample image not found: {image_path}")
                sample_errors += 1
                continue
            if not os.path.exists(mask_path):
                print(f"Warning: Sample mask not found: {mask_path}")
                sample_errors += 1
                continue
            
            image = Image.open(image_path).convert("RGB")
            true_mask = np.array(Image.open(mask_path))

            # Try to load from cache first
            pred_mask = None
            if use_cache and cache_dir and model_path:
                cache_key = get_cache_key(model_path, img_name)
                pred_mask, _, _ = load_prediction_cache(cache_dir, cache_key)

            # Get prediction if not cached
            if pred_mask is None:
                pred_mask = predict_single_image(
                    model, image, device, transform, return_time=False)
                
                # Save to cache if enabled
                if use_cache and cache_dir and model_path:
                    cache_key = get_cache_key(model_path, img_name)
                    # We don't have timing here, so save with None
                    save_prediction_cache(cache_dir, cache_key, pred_mask, None)

            # Calculate metrics for this sample
            iou = calculate_iou(pred_mask, true_mask)
            dice = calculate_dice_coefficient(pred_mask, true_mask)

            # Plot original image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Original Image\n{img_name}')
            axes[i, 0].axis('off')

            # Plot ground truth mask
            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')

            # Plot predicted mask
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')

            # Plot overlay
            overlay = np.array(image)
            # Create colored overlays
            true_overlay = overlay.copy()
            pred_overlay = overlay.copy()

            # Green for ground truth, Red for prediction
            true_mask_binary = (true_mask > 127).astype(np.uint8)
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)

            true_overlay[true_mask_binary > 0] = [0, 255, 0]  # Green
            pred_overlay[pred_mask_binary > 0] = [255, 0, 0]  # Red

            # Combine overlays
            combined_overlay = overlay.copy()
            combined_overlay[true_mask_binary > 0] = [0, 255, 0]  # Green for GT
            combined_overlay[pred_mask_binary > 0] = [
                255, 0, 0]  # Red for prediction
            # Yellow for overlap
            overlap = np.logical_and(true_mask_binary, pred_mask_binary)
            combined_overlay[overlap] = [255, 255, 0]  # Yellow

            axes[i, 3].imshow(combined_overlay)
            axes[i, 3].set_title(
                f'Overlay (GT=Green, Pred=Red)\nIoU: {iou:.3f}, Dice: {dice:.3f}')
            axes[i, 3].axis('off')
            
        except Exception as e:
            print(f"Error creating sample prediction plot for sample {i} ({img_name}): {e}")
            sample_errors += 1
            continue
    
    if sample_errors > 0:
        print(f"Sample predictions plot completed with {sample_errors} errors")

    plt.tight_layout()
    plt.savefig(os.path.join(eval_plot_dir, 'sample_predictions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f'Sample predictions plot saved to {eval_plot_dir}/sample_predictions.png')


#############################################################################
# Main Evaluation Function
#############################################################################

def print_evaluation_summary(accuracy_results, speed_results):
    """
    Print a comprehensive evaluation summary.

    Args:
        accuracy_results (dict): Results from accuracy evaluation
        speed_results (dict): Results from speed benchmarking
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)

    print(f"\nACCURACY METRICS (n={accuracy_results['num_samples']} samples):")
    print(f"  IoU Score:")
    print(
        f"    Mean: {accuracy_results['mean_iou']:.4f} ± {accuracy_results['std_iou']:.4f}")
    print(f"    Median: {accuracy_results['median_iou']:.4f}")

    print(f"  Dice Coefficient:")
    print(
        f"    Mean: {accuracy_results['mean_dice']:.4f} ± {accuracy_results['std_dice']:.4f}")
    print(f"    Median: {accuracy_results['median_dice']:.4f}")

    print(f"  Pixel Accuracy:")
    print(
        f"    Mean: {accuracy_results['mean_pixel_accuracy']:.4f} ± {accuracy_results['std_pixel_accuracy']:.4f}")
    print(f"    Median: {accuracy_results['median_pixel_accuracy']:.4f}")

    print(f"\nCONFIDENCE METRICS:")
    print(f"  Confidence in Vessel Regions:")
    print(
        f"    Mean: {accuracy_results['mean_confidence_in_vessels']:.4f} ± {accuracy_results['std_confidence_in_vessels']:.4f}")
    print(
        f"    Median: {accuracy_results['median_confidence_in_vessels']:.4f}")

    print(f"  Confidence in Background Regions:")
    print(
        f"    Mean: {accuracy_results['mean_confidence_in_background']:.4f} ± {accuracy_results['std_confidence_in_background']:.4f}")
    print(
        f"    Median: {accuracy_results['median_confidence_in_background']:.4f}")

    print(f"  Confidence Separation (Vessel - Background):")
    print(
        f"    Mean: {accuracy_results['mean_confidence_separation']:.4f} ± {accuracy_results['std_confidence_separation']:.4f}")
    print(
        f"    Median: {accuracy_results['median_confidence_separation']:.4f}")

    print(f"\nSPEED METRICS (n={speed_results['num_samples']} samples):")
    print(f"  Prediction Time:")
    print(
        f"    Mean: {speed_results['mean_time']:.4f} ± {speed_results['std_time']:.4f} seconds")
    print(f"    Median: {speed_results['median_time']:.4f} seconds")
    print(
        f"    Range: {speed_results['min_time']:.4f} - {speed_results['max_time']:.4f} seconds")

    print(f"  Frames Per Second:")
    print(f"    Mean: {speed_results['mean_fps']:.2f} FPS")
    print(f"    Median: {speed_results['median_fps']:.2f} FPS")

    print("\n" + "="*60)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Mask R-CNN model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model (default: best_val_mask_rcnn_model.pth)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--warmup-samples', type=int, default=10,
                        help='Number of warmup samples before timing')
    parser.add_argument('--no-save-results', action='store_true',
                        help='Skip saving detailed results to files')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable prediction caching')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear prediction cache before running')
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    print("="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)

    # Setup device
    print("Setting up compute device...")
    device = setup_device()
    print(f'Using device: {device}')

    # Load data
    print("\nLoading validation data...")
    try:
        labelled_data_dir, img_dir, mask_dir, image_names, mask_names = load_data(
            data_dir)
        # train_imgs, train_masks, val_imgs, val_masks = split_train_val(
        #     image_names, mask_names)
        val_imgs = image_names
        val_masks = mask_names
        print(f"Loaded {len(val_imgs)} validation images and {len(val_masks)} masks")
        
        if len(val_imgs) == 0:
            print("Error: No validation images found!")
            return
            
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return

    # Setup augmented directories (may be empty)
    aug_img_dir = os.path.join(labelled_data_dir, 'augmented_images/')
    aug_mask_dir = os.path.join(labelled_data_dir, 'augmented_masks/')
    
    print(f"Image directory: {img_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Augmented image directory: {aug_img_dir}")
    print(f"Augmented mask directory: {aug_mask_dir}")

    # Load model
    print("\nLoading trained model...")
    try:
        model, device, transform = load_model(args.model_path, device)
        model_path = args.model_path if args.model_path else os.path.join(
            artifacts_dir, 'best_val_mask_rcnn_model.pth')
        print(f"Model successfully loaded from: {model_path}")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please train a model first or specify a valid model path.")
        return
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return

    # Handle cache options
    use_cache = not args.no_cache
    if args.clear_cache:
        print("\nClearing prediction cache...")
        cache_dir = os.path.join(artifacts_dir, 'prediction_cache')
        clear_prediction_cache(cache_dir)
    
    if use_cache:
        cache_dir = os.path.join(artifacts_dir, 'prediction_cache')
        cache_info = get_cache_info(cache_dir)
        print(f"Prediction caching enabled. Current cache: {cache_info['num_files']} files, {cache_info['total_size_mb']:.1f} MB")
    else:
        print("Prediction caching disabled.")

    # Evaluate accuracy and speed together
    print(f"\n{'='*60}")
    print("RUNNING MODEL EVALUATION")
    print(f"{'='*60}")
    
    try:
        output_dir = os.path.join(data_dir, 'plots', 'prediction_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Prediction visualizations will be saved to: {output_dir}")
        
        accuracy_results, speed_results = evaluate_model_accuracy_and_speed(
            model, val_imgs, val_masks, img_dir, mask_dir,
            aug_img_dir, aug_mask_dir, device, transform,
            output_dir,
            max_samples=args.max_samples,
            num_warmup=args.warmup_samples,
            use_cache=use_cache,
            model_path=model_path
        )
        
        print("Model evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    # Print summary
    print_evaluation_summary(accuracy_results, speed_results)

    # Create visualizations
    print(f"\n{'='*60}")
    print("CREATING EVALUATION VISUALIZATIONS")
    print(f"{'='*60}")
    
    try:
        plot_evaluation_results(accuracy_results, speed_results, plot_dir)
        plot_metrics_vs_positive_area(accuracy_results, plot_dir)
        create_sample_predictions_plot(
            model, val_imgs, val_masks, img_dir, mask_dir,
            aug_img_dir, aug_mask_dir, device, transform, plot_dir,
            use_cache=use_cache, model_path=model_path
        )
        print("All evaluation plots created successfully!")
        
    except Exception as e:
        print(f"Error creating evaluation plots: {e}")

    # Save detailed results unless disabled
    if not args.no_save_results:
        print(f"\n{'='*60}")
        print("SAVING DETAILED RESULTS")
        print(f"{'='*60}")
        
        try:
            results_dir = os.path.join(artifacts_dir, 'evaluation_results')
            os.makedirs(results_dir, exist_ok=True)

            # Save accuracy results
            print("Saving accuracy metrics...")
            np.save(os.path.join(results_dir, 'iou_scores.npy'),
                    accuracy_results['iou_scores'])
            np.save(os.path.join(results_dir, 'dice_scores.npy'),
                    accuracy_results['dice_scores'])
            np.save(os.path.join(results_dir, 'pixel_accuracies.npy'),
                    accuracy_results['pixel_accuracies'])
            np.save(os.path.join(results_dir, 'prediction_times.npy'),
                    accuracy_results['prediction_times'])
            np.save(os.path.join(results_dir, 'positive_areas.npy'),
                    accuracy_results['positive_areas'])

            # Save confidence results
            print("Saving confidence metrics...")
            np.save(os.path.join(results_dir, 'confidence_in_vessels.npy'),
                    accuracy_results['confidence_in_vessels'])
            np.save(os.path.join(results_dir, 'confidence_in_background.npy'),
                    accuracy_results['confidence_in_background'])
            np.save(os.path.join(results_dir, 'confidence_separations.npy'),
                    accuracy_results['confidence_separations'])

            # Save per-image confidence metrics as a CSV-like format
            print("Saving per-image confidence metrics...")
            import json
            with open(os.path.join(results_dir, 'per_image_confidence_metrics.json'), 'w') as f:
                json.dump(
                    accuracy_results['per_image_confidence_metrics'], f, indent=2)

            # Save speed results
            print("Saving speed benchmark results...")
            np.save(os.path.join(results_dir, 'speed_benchmark_times.npy'),
                    speed_results['times'])

            print(f"All detailed results saved successfully to: {results_dir}")
            
        except Exception as e:
            print(f"Error saving detailed results: {e}")

    print(f"\n{'='*60}")
    print("MODEL EVALUATION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
