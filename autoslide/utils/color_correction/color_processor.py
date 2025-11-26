"""
Unified color correction and normalization module with backup/restore functionality.

This module merges color correction (Reinhard method) and histogram percentile
normalization into a single interface with support for:
1. Backing up original images before processing
2. Replacing images with color-corrected versions
3. Restoring original images from backups
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import logging
import shutil
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ColorProcessor:
    """
    Unified color correction and normalization processor.
    
    Supports multiple methods:
    - 'reinhard': Reinhard et al. color transfer in LAB space
    - 'histogram': Histogram matching in LAB space
    - 'percentile': Histogram percentile normalization
    """

    def __init__(
        self,
        reference_images: Union[str, Path, List[Union[str, Path]]],
        method: str = 'reinhard',
        percentiles: Tuple[float, float] = (1.0, 99.0)
    ):
        """
        Initialize the color processor.

        Args:
            reference_images: Path(s) to reference image(s) from the original dataset
            method: Processing method ('reinhard', 'histogram', 'percentile')
            percentiles: Percentile range for percentile method (low, high)
        """
        if isinstance(reference_images, (str, Path)):
            reference_images = [reference_images]

        self.reference_images = [Path(img) for img in reference_images]
        self.method = method
        self.percentiles = percentiles
        self.reference_stats = None
        self._compute_reference_statistics()

    def _compute_reference_statistics(self):
        """Compute statistics from reference images based on method."""
        if self.method in ['reinhard', 'histogram']:
            self._compute_lab_statistics()
        elif self.method == 'percentile':
            self._compute_percentile_statistics()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_lab_statistics(self):
        """Compute LAB color space statistics for Reinhard/histogram methods."""
        all_pixels = []

        for img_path in self.reference_images:
            if not img_path.exists():
                logger.warning(f"Reference image not found: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load reference image: {img_path}")
                continue

            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            all_pixels.append(img_lab.reshape(-1, 3))

        if not all_pixels:
            raise ValueError("No valid reference images found")

        combined_pixels = np.vstack(all_pixels)
        self.reference_stats = {
            'mean': np.mean(combined_pixels, axis=0),
            'std': np.std(combined_pixels, axis=0)
        }

        logger.info(
            f"Computed LAB statistics from {len(self.reference_images)} images")

    def _compute_percentile_statistics(self):
        """Compute percentile statistics for percentile normalization."""
        all_pixels = {0: [], 1: [], 2: []}

        for img_path in self.reference_images:
            if not img_path.exists():
                logger.warning(f"Reference image not found: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load reference image: {img_path}")
                continue

            for channel in range(3):
                all_pixels[channel].append(img[:, :, channel].flatten())

        if not all_pixels[0]:
            raise ValueError("No valid reference images found")

        self.reference_stats = {}
        for channel in range(3):
            combined = np.concatenate(all_pixels[channel])
            low_val = np.percentile(combined, self.percentiles[0])
            high_val = np.percentile(combined, self.percentiles[1])
            self.reference_stats[channel] = (low_val, high_val)

        logger.info(
            f"Computed percentile statistics from {len(self.reference_images)} images")

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color processing to an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Processed image as numpy array (BGR format)
        """
        if self.method == 'reinhard':
            return self._reinhard_color_transfer(image)
        elif self.method == 'histogram':
            return self._histogram_matching(image)
        elif self.method == 'percentile':
            return self._percentile_normalization(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _reinhard_color_transfer(self, image: np.ndarray) -> np.ndarray:
        """Apply Reinhard et al. color transfer in LAB space."""
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        img_mean = np.mean(img_lab.reshape(-1, 3), axis=0)
        img_std = np.std(img_lab.reshape(-1, 3), axis=0)
        img_std = np.where(img_std == 0, 1, img_std)

        img_lab -= img_mean
        scale = self.reference_stats['std'] / img_std
        img_lab *= scale
        img_lab += self.reference_stats['mean']

        img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    def _histogram_matching(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram matching in LAB space."""
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        matched_channels = []
        for channel_idx in range(3):
            channel = img_lab[:, :, channel_idx]
            matched = self._match_histogram_channel(
                channel,
                self.reference_stats['mean'][channel_idx],
                self.reference_stats['std'][channel_idx]
            )
            matched_channels.append(matched)

        matched_lab = np.stack(matched_channels, axis=2)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    def _match_histogram_channel(
        self,
        channel: np.ndarray,
        target_mean: float,
        target_std: float
    ) -> np.ndarray:
        """Match histogram of a single channel to target statistics."""
        current_mean = np.mean(channel)
        current_std = np.std(channel)

        if current_std == 0:
            current_std = 1

        normalized = (channel.astype(np.float32) - current_mean) / current_std
        matched = normalized * target_std + target_mean
        return np.clip(matched, 0, 255).astype(np.uint8)

    def _percentile_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram percentile normalization."""
        normalized = np.zeros_like(image, dtype=np.float32)

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float32)

            low_in = np.percentile(channel_data, self.percentiles[0])
            high_in = np.percentile(channel_data, self.percentiles[1])

            low_ref, high_ref = self.reference_stats[channel]

            if high_in - low_in < 1e-6:
                normalized[:, :, channel] = channel_data
                continue

            normalized[:, :, channel] = (
                (channel_data - low_in) / (high_in - low_in) *
                (high_ref - low_ref) + low_ref
            )

        return np.clip(normalized, 0, 255).astype(np.uint8)

    def process_image_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Process an image file.

        Args:
            input_path: Path to input image
            output_path: Path to save processed image

        Returns:
            True if successful, False otherwise
        """
        try:
            img = cv2.imread(str(input_path))
            if img is None:
                logger.error(f"Failed to load image: {input_path}")
                return False

            processed = self.process_image(img)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)

            logger.info(f"Processed image saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing image {input_path}: {e}")
            return False


def backup_images(
    image_dir: Union[str, Path],
    backup_dir: Union[str, Path],
    file_pattern: str = '*.png',
    metadata: Optional[Dict] = None
) -> Tuple[int, int]:
    """
    Backup images from a directory before processing.

    Args:
        image_dir: Directory containing images to backup
        backup_dir: Directory to store backups
        file_pattern: Glob pattern for image files
        metadata: Optional metadata to store with backup

    Returns:
        Tuple of (successful_count, failed_count)
    """
    image_dir = Path(image_dir)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Store metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'backup_timestamp': datetime.now().isoformat(),
        'source_directory': str(image_dir.absolute()),
        'file_pattern': file_pattern
    })
    
    metadata_path = backup_dir / 'backup_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Backup images
    image_files = list(image_dir.glob(file_pattern))
    successful = 0
    failed = 0

    logger.info(f"Backing up {len(image_files)} images to {backup_dir}")

    for img_path in image_files:
        try:
            backup_path = backup_dir / img_path.name
            shutil.copy2(img_path, backup_path)
            successful += 1
        except Exception as e:
            logger.error(f"Failed to backup {img_path}: {e}")
            failed += 1

    logger.info(f"Backup complete: {successful} successful, {failed} failed")
    return successful, failed


def restore_originals(
    backup_dir: Union[str, Path],
    target_dir: Optional[Union[str, Path]] = None,
    verify: bool = True
) -> Tuple[int, int]:
    """
    Restore original images from backup.

    Args:
        backup_dir: Directory containing backed up images
        target_dir: Directory to restore images to (defaults to original location)
        verify: Whether to verify backup metadata before restoring

    Returns:
        Tuple of (successful_count, failed_count)
    """
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        logger.error(f"Backup directory not found: {backup_dir}")
        return 0, 0

    # Load metadata
    metadata_path = backup_dir / 'backup_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if target_dir is None:
            target_dir = Path(metadata['source_directory'])
        
        logger.info(f"Restoring from backup created at {metadata['backup_timestamp']}")
    else:
        if target_dir is None:
            logger.error("No metadata found and no target directory specified")
            return 0, 0
        logger.warning("No backup metadata found")

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Restore images (excluding metadata file)
    backup_files = [f for f in backup_dir.iterdir() 
                   if f.is_file() and f.name != 'backup_metadata.json']
    successful = 0
    failed = 0

    logger.info(f"Restoring {len(backup_files)} images to {target_dir}")

    for backup_path in backup_files:
        try:
            target_path = target_dir / backup_path.name
            shutil.copy2(backup_path, target_path)
            successful += 1
            logger.debug(f"Restored {backup_path.name}")
        except Exception as e:
            logger.error(f"Failed to restore {backup_path}: {e}")
            failed += 1

    logger.info(f"Restore complete: {successful} successful, {failed} failed")
    return successful, failed


def list_backups(backup_root: Union[str, Path]) -> List[Dict]:
    """
    List all available backups in a directory.

    Args:
        backup_root: Root directory containing backup subdirectories

    Returns:
        List of dictionaries with backup information
    """
    backup_root = Path(backup_root)
    
    if not backup_root.exists():
        logger.warning(f"Backup root directory not found: {backup_root}")
        return []

    backups = []
    
    for backup_dir in backup_root.iterdir():
        if not backup_dir.is_dir():
            continue
            
        metadata_path = backup_dir / 'backup_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Count backup files
            file_count = len([f for f in backup_dir.iterdir() 
                            if f.is_file() and f.name != 'backup_metadata.json'])
            
            backups.append({
                'backup_dir': str(backup_dir),
                'timestamp': metadata.get('backup_timestamp', 'unknown'),
                'source_directory': metadata.get('source_directory', 'unknown'),
                'file_count': file_count,
                'method': metadata.get('method', 'unknown')
            })

    return sorted(backups, key=lambda x: x['timestamp'], reverse=True)


def batch_process_directory(
    input_dir: Union[str, Path],
    reference_images: Union[str, Path, List[Union[str, Path]]],
    method: str = 'reinhard',
    percentiles: Tuple[float, float] = (1.0, 99.0),
    file_pattern: str = '*.png',
    backup: bool = True,
    backup_root: Optional[Union[str, Path]] = None,
    replace_originals: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, any]:
    """
    Process all images in a directory with optional backup.

    Args:
        input_dir: Directory containing images to process
        reference_images: Path(s) to reference image(s)
        method: Processing method ('reinhard', 'histogram', 'percentile')
        percentiles: Percentile range for percentile method
        file_pattern: Glob pattern for image files
        backup: Whether to backup original images
        backup_root: Root directory for backups (defaults to input_dir/backups)
        replace_originals: Whether to replace original images (True) or save to output_dir (False)
        output_dir: Output directory if not replacing originals

    Returns:
        Dictionary with processing results and backup information
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return {'success': False, 'error': 'Input directory not found'}

    # Setup backup
    backup_dir = None
    if backup:
        if backup_root is None:
            backup_root = input_dir / 'backups'
        else:
            backup_root = Path(backup_root)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = backup_root / f'backup_{timestamp}'
        
        metadata = {
            'method': method,
            'percentiles': percentiles if method == 'percentile' else None,
            'file_pattern': file_pattern,
            'replace_originals': replace_originals
        }
        
        backup_success, backup_failed = backup_images(
            input_dir, backup_dir, file_pattern, metadata)
        
        if backup_failed > 0:
            logger.warning(f"Some files failed to backup: {backup_failed}")

    # Setup output directory
    if replace_originals:
        output_dir = input_dir
    else:
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_processed"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = ColorProcessor(reference_images, method, percentiles)

    # Process images
    image_files = list(input_dir.glob(file_pattern))
    successful = 0
    failed = 0

    logger.info(f"Processing {len(image_files)} images with method '{method}'")

    for img_path in image_files:
        output_path = output_dir / img_path.name
        if processor.process_image_file(img_path, output_path):
            successful += 1
        else:
            failed += 1

    result = {
        'success': True,
        'processed': successful,
        'failed': failed,
        'method': method,
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'replace_originals': replace_originals
    }

    if backup:
        result['backup_dir'] = str(backup_dir)
        result['backup_count'] = backup_success

    logger.info(
        f"Processing complete: {successful} successful, {failed} failed")

    return result
