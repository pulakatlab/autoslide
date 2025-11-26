"""
Color correction preprocessing module for histological images.

This module provides functionality to correct color variations in new datasets
by transferring color characteristics from reference images in the original dataset.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ColorCorrector:
    """
    Color correction using various color transfer methods.

    This class implements color correction for histological images by transferring
    color characteristics from reference images to target images. This helps
    normalize color variations across different datasets or scanning sessions.
    """

    def __init__(self, reference_images: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the ColorCorrector with reference images.

        Args:
            reference_images: Path(s) to reference image(s) from the original dataset.
                             Can be a single path or a list of paths.
        """
        if isinstance(reference_images, (str, Path)):
            reference_images = [reference_images]

        self.reference_images = [Path(img) for img in reference_images]
        self.reference_stats = None
        self._compute_reference_statistics()

    def _compute_reference_statistics(self):
        """Compute color statistics from reference images."""
        all_pixels = []

        for img_path in self.reference_images:
            if not img_path.exists():
                logger.warning(f"Reference image not found: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load reference image: {img_path}")
                continue

            # Convert BGR to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            all_pixels.append(img_lab.reshape(-1, 3))

        if not all_pixels:
            raise ValueError("No valid reference images found")

        # Combine all pixels from reference images
        combined_pixels = np.vstack(all_pixels)

        # Compute mean and standard deviation for each channel
        self.reference_stats = {
            'mean': np.mean(combined_pixels, axis=0),
            'std': np.std(combined_pixels, axis=0)
        }

        logger.info(
            f"Computed reference statistics from {len(self.reference_images)} images")

    def correct_image(self, image: np.ndarray, method: str = 'reinhard') -> np.ndarray:
        """
        Apply color correction to an image.

        Args:
            image: Input image as numpy array (BGR format)
            method: Color correction method to use. Options:
                   - 'reinhard': Reinhard et al. color transfer method
                   - 'histogram': Histogram matching

        Returns:
            Color-corrected image as numpy array (BGR format)
        """
        if method == 'reinhard':
            return self._reinhard_color_transfer(image)
        elif method == 'histogram':
            return self._histogram_matching(image)
        else:
            raise ValueError(f"Unknown color correction method: {method}")

    def _reinhard_color_transfer(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Reinhard et al. color transfer method.

        Based on: "Color Transfer between Images" by Reinhard et al., 2001
        This method transfers color characteristics in LAB color space.
        """
        # Convert to LAB color space
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute statistics of the input image
        img_mean = np.mean(img_lab.reshape(-1, 3), axis=0)
        img_std = np.std(img_lab.reshape(-1, 3), axis=0)

        # Avoid division by zero
        img_std = np.where(img_std == 0, 1, img_std)

        # Apply color transfer
        # 1. Subtract mean
        img_lab -= img_mean

        # 2. Scale by standard deviation ratio
        scale = self.reference_stats['std'] / img_std
        img_lab *= scale

        # 3. Add reference mean
        img_lab += self.reference_stats['mean']

        # Clip values to valid range
        img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)

        # Convert back to BGR
        result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        return result

    def _histogram_matching(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram matching for color correction.

        This method matches the histogram of the input image to the
        reference histogram in LAB color space.
        """
        # Convert to LAB color space
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Match histogram for each channel
        matched_channels = []
        for channel_idx in range(3):
            channel = img_lab[:, :, channel_idx]
            matched = self._match_histogram_channel(
                channel,
                self.reference_stats['mean'][channel_idx],
                self.reference_stats['std'][channel_idx]
            )
            matched_channels.append(matched)

        # Combine channels
        matched_lab = np.stack(matched_channels, axis=2)

        # Convert back to BGR
        result = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

        return result

    def _match_histogram_channel(
        self,
        channel: np.ndarray,
        target_mean: float,
        target_std: float
    ) -> np.ndarray:
        """Match histogram of a single channel to target statistics."""
        # Compute current statistics
        current_mean = np.mean(channel)
        current_std = np.std(channel)

        # Avoid division by zero
        if current_std == 0:
            current_std = 1

        # Normalize and scale
        normalized = (channel.astype(np.float32) - current_mean) / current_std
        matched = normalized * target_std + target_mean

        # Clip to valid range
        matched = np.clip(matched, 0, 255).astype(np.uint8)

        return matched

    def correct_image_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        method: str = 'reinhard'
    ) -> bool:
        """
        Apply color correction to an image file.

        Args:
            input_path: Path to input image
            output_path: Path to save corrected image
            method: Color correction method to use

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            img = cv2.imread(str(input_path))
            if img is None:
                logger.error(f"Failed to load image: {input_path}")
                return False

            # Apply correction
            corrected = self.correct_image(img, method=method)

            # Save result
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), corrected)

            logger.info(f"Color corrected image saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error correcting image {input_path}: {e}")
            return False

    def get_color_statistics(self, image: np.ndarray) -> dict:
        """
        Compute color statistics for an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary containing mean and std for each LAB channel
        """
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = img_lab.reshape(-1, 3)

        return {
            'mean': np.mean(pixels, axis=0),
            'std': np.std(pixels, axis=0)
        }


def compute_color_divergence(
    image1: np.ndarray,
    image2: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute color distribution divergence between two images.

    This function measures how different the color distributions are between
    two images, which can be used to detect color variations in datasets.

    Args:
        image1: First image as numpy array (BGR format)
        image2: Second image as numpy array (BGR format)
        metric: Distance metric to use ('euclidean', 'manhattan', 'kl_divergence')

    Returns:
        Divergence score (lower means more similar)
    """
    # Convert both images to LAB
    img1_lab = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    # Compute statistics
    mean1 = np.mean(img1_lab.reshape(-1, 3), axis=0)
    std1 = np.std(img1_lab.reshape(-1, 3), axis=0)
    mean2 = np.mean(img2_lab.reshape(-1, 3), axis=0)
    std2 = np.std(img2_lab.reshape(-1, 3), axis=0)

    if metric == 'euclidean':
        # Euclidean distance between mean and std vectors
        mean_dist = np.linalg.norm(mean1 - mean2)
        std_dist = np.linalg.norm(std1 - std2)
        return mean_dist + std_dist

    elif metric == 'manhattan':
        # Manhattan distance
        mean_dist = np.sum(np.abs(mean1 - mean2))
        std_dist = np.sum(np.abs(std1 - std2))
        return mean_dist + std_dist

    elif metric == 'kl_divergence':
        # Simplified KL divergence approximation using Gaussian assumption
        # KL(P||Q) = log(σ2/σ1) + (σ1^2 + (μ1-μ2)^2)/(2σ2^2) - 1/2
        kl_div = 0
        for i in range(3):
            if std2[i] == 0:
                std2[i] = 1e-6
            kl_div += np.log(std2[i] / (std1[i] + 1e-6))
            kl_div += (std1[i]**2 + (mean1[i] - mean2[i])
                       ** 2) / (2 * std2[i]**2)
            kl_div -= 0.5
        return kl_div

    else:
        raise ValueError(f"Unknown metric: {metric}")


def batch_color_correction(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reference_images: Union[str, Path, List[Union[str, Path]]],
    method: str = 'reinhard',
    file_pattern: str = '*.png'
) -> Tuple[int, int]:
    """
    Apply color correction to all images in a directory.

    Args:
        input_dir: Directory containing images to correct
        output_dir: Directory to save corrected images
        reference_images: Path(s) to reference image(s)
        method: Color correction method to use
        file_pattern: Glob pattern for image files

    Returns:
        Tuple of (successful_count, failed_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize corrector
    corrector = ColorCorrector(reference_images)

    # Process all matching files
    image_files = list(input_dir.glob(file_pattern))
    successful = 0
    failed = 0

    logger.info(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        output_path = output_dir / img_path.name
        if corrector.correct_image_file(img_path, output_path, method=method):
            successful += 1
        else:
            failed += 1

    logger.info(
        f"Batch correction complete: {successful} successful, {failed} failed")

    return successful, failed
