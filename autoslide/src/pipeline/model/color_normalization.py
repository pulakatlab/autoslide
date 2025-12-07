"""
Color normalization using histogram percentile transformation.

This module provides histogram-based color normalization for histological images,
particularly useful for normalizing staining variations across different batches
or scanning sessions.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HistogramPercentileNormalizer:
    """
    Histogram percentile-based color normalization.

    This class implements color normalization by matching histogram percentiles
    between reference and target images. This method is particularly effective
    for histological images where staining intensity can vary.
    """

    def __init__(
        self,
        reference_images: Union[str, Path, List[Union[str, Path]]],
        percentiles: Tuple[float, float] = (1.0, 99.0)
    ):
        """
        Initialize the normalizer with reference images.

        Args:
            reference_images: Path(s) to reference image(s) from the original dataset.
                             Can be a single path or a list of paths.
            percentiles: Tuple of (low, high) percentiles to use for normalization.
                        Default (1.0, 99.0) clips extreme values.
        """
        if isinstance(reference_images, (str, Path)):
            reference_images = [reference_images]

        self.reference_images = [Path(img) for img in reference_images]
        self.percentiles = percentiles
        self.reference_percentiles = None
        self._compute_reference_percentiles()

    def _compute_reference_percentiles(self):
        """Compute percentile values from reference images."""
        all_pixels = {0: [], 1: [], 2: []}  # BGR channels

        for img_path in self.reference_images:
            if not img_path.exists():
                logger.warning(f"Reference image not found: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load reference image: {img_path}")
                continue

            # Collect pixels for each channel
            for channel in range(3):
                all_pixels[channel].append(img[:, :, channel].flatten())

        if not all_pixels[0]:
            raise ValueError("No valid reference images found")

        # Combine all pixels and compute percentiles for each channel
        self.reference_percentiles = {}
        for channel in range(3):
            combined = np.concatenate(all_pixels[channel])
            low_val = np.percentile(combined, self.percentiles[0])
            high_val = np.percentile(combined, self.percentiles[1])
            self.reference_percentiles[channel] = (low_val, high_val)

        logger.info(
            f"Computed reference percentiles from {len(self.reference_images)} images"
        )
        logger.info(
            f"Percentile range: {self.percentiles[0]}% - {self.percentiles[1]}%")

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram percentile normalization to an image.

        This method:
        1. Computes percentiles for each channel in the input image
        2. Maps the percentile range to match the reference percentiles
        3. Clips values outside the percentile range

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Normalized image as numpy array (BGR format)
        """
        normalized = np.zeros_like(image, dtype=np.float32)

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float32)

            # Compute percentiles for this channel
            low_in = np.percentile(channel_data, self.percentiles[0])
            high_in = np.percentile(channel_data, self.percentiles[1])

            # Get reference percentiles
            low_ref, high_ref = self.reference_percentiles[channel]

            # Avoid division by zero
            if high_in - low_in < 1e-6:
                normalized[:, :, channel] = channel_data
                continue

            # Linear mapping from input percentile range to reference percentile range
            # Formula: out = (in - low_in) / (high_in - low_in) * (high_ref - low_ref) + low_ref
            normalized[:, :, channel] = (
                (channel_data - low_in) / (high_in - low_in) *
                (high_ref - low_ref) + low_ref
            )

        # Clip to valid range and convert to uint8
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized

    def normalize_image_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Apply normalization to an image file.

        Args:
            input_path: Path to input image
            output_path: Path to save normalized image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            img = cv2.imread(str(input_path))
            if img is None:
                logger.error(f"Failed to load image: {input_path}")
                return False

            # Apply normalization
            normalized = self.normalize_image(img)

            # Save result
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), normalized)

            logger.info(f"Normalized image saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error normalizing image {input_path}: {e}")
            return False

    def get_image_percentiles(self, image: np.ndarray) -> dict:
        """
        Compute percentile values for an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary with percentile values for each channel
        """
        percentiles_dict = {}
        for channel in range(3):
            channel_data = image[:, :, channel]
            low_val = np.percentile(channel_data, self.percentiles[0])
            high_val = np.percentile(channel_data, self.percentiles[1])
            percentiles_dict[f'channel_{channel}'] = (low_val, high_val)

        return percentiles_dict


def batch_histogram_normalization(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reference_images: Union[str, Path, List[Union[str, Path]]],
    percentiles: Tuple[float, float] = (1.0, 99.0),
    file_pattern: str = '*.png'
) -> Tuple[int, int]:
    """
    Apply histogram percentile normalization to all images in a directory.

    Args:
        input_dir: Directory containing images to normalize
        output_dir: Directory to save normalized images
        reference_images: Path(s) to reference image(s)
        percentiles: Tuple of (low, high) percentiles for normalization
        file_pattern: Glob pattern for image files

    Returns:
        Tuple of (successful_count, failed_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize normalizer
    normalizer = HistogramPercentileNormalizer(reference_images, percentiles)

    # Process all matching files
    image_files = list(input_dir.glob(file_pattern))
    successful = 0
    failed = 0

    logger.info(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        output_path = output_dir / img_path.name
        if normalizer.normalize_image_file(img_path, output_path):
            successful += 1
        else:
            failed += 1

    logger.info(
        f"Batch normalization complete: {successful} successful, {failed} failed")

    return successful, failed


def compare_histograms(
    image1: np.ndarray,
    image2: np.ndarray,
    bins: int = 256
) -> dict:
    """
    Compare histograms of two images.

    Args:
        image1: First image as numpy array (BGR format)
        image2: Second image as numpy array (BGR format)
        bins: Number of histogram bins

    Returns:
        Dictionary with histogram comparison metrics
    """
    metrics = {}

    for channel in range(3):
        hist1 = cv2.calcHist([image1], [channel], None, [bins], [0, 256])
        hist2 = cv2.calcHist([image2], [channel], None, [bins], [0, 256])

        # Normalize histograms
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Compute correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Compute chi-square distance
        chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

        # Compute intersection
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

        metrics[f'channel_{channel}'] = {
            'correlation': correlation,
            'chi_square': chi_square,
            'intersection': intersection
        }

    return metrics
