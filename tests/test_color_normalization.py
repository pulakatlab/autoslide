"""
Tests for histogram percentile color normalization module.

This module tests:
- Histogram percentile normalization functionality
- Batch processing
- Histogram comparison utilities
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from autoslide.pipeline.model.color_normalization import (
    HistogramPercentileNormalizer,
    batch_histogram_normalization,
    compare_histograms
)


class TestHistogramPercentileNormalizer:
    """Test suite for HistogramPercentileNormalizer class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # Blue channel
        img[:, :, 1] = 150  # Green channel
        img[:, :, 2] = 200  # Red channel
        return img
    
    @pytest.fixture
    def reference_image(self):
        """Create a reference image with different intensity distribution."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 80   # Blue channel - darker
        img[:, :, 1] = 140  # Green channel - darker
        img[:, :, 2] = 220  # Red channel - brighter
        return img
    
    @pytest.fixture
    def temp_reference_file(self, reference_image):
        """Create a temporary reference image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, reference_image)
            yield f.name
        Path(f.name).unlink()
    
    def test_normalizer_initialization(self, temp_reference_file):
        """Test normalizer initialization with reference images."""
        normalizer = HistogramPercentileNormalizer(temp_reference_file)
        assert normalizer.reference_percentiles is not None
        assert len(normalizer.reference_percentiles) == 3
        for channel in range(3):
            assert channel in normalizer.reference_percentiles
            low, high = normalizer.reference_percentiles[channel]
            assert low < high
    
    def test_normalizer_multiple_references(self, temp_reference_file):
        """Test normalizer with multiple reference images."""
        normalizer = HistogramPercentileNormalizer(
            [temp_reference_file, temp_reference_file]
        )
        assert normalizer.reference_percentiles is not None
        assert len(normalizer.reference_images) == 2
    
    def test_custom_percentiles(self, temp_reference_file):
        """Test normalizer with custom percentile range."""
        normalizer = HistogramPercentileNormalizer(
            temp_reference_file,
            percentiles=(5.0, 95.0)
        )
        assert normalizer.percentiles == (5.0, 95.0)
        assert normalizer.reference_percentiles is not None
    
    def test_normalize_image(self, sample_image, temp_reference_file):
        """Test image normalization."""
        normalizer = HistogramPercentileNormalizer(temp_reference_file)
        normalized = normalizer.normalize_image(sample_image)
        
        # Check output shape and type
        assert normalized.shape == sample_image.shape
        assert normalized.dtype == np.uint8
        
        # Check that normalization was applied
        assert not np.array_equal(normalized, sample_image)
    
    def test_normalize_preserves_structure(self, temp_reference_file):
        """Test that normalization preserves image structure."""
        # Create image with distinct regions
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75, :] = 200  # Bright square in center
        img[0:25, 0:25, :] = 50     # Dark square in corner
        
        normalizer = HistogramPercentileNormalizer(temp_reference_file)
        normalized = normalizer.normalize_image(img)
        
        # Check that bright region is still brighter than dark region
        center_mean = np.mean(normalized[25:75, 25:75, :])
        corner_mean = np.mean(normalized[0:25, 0:25, :])
        assert center_mean > corner_mean
    
    def test_get_image_percentiles(self, sample_image, temp_reference_file):
        """Test percentile computation for an image."""
        normalizer = HistogramPercentileNormalizer(temp_reference_file)
        percentiles = normalizer.get_image_percentiles(sample_image)
        
        assert 'channel_0' in percentiles
        assert 'channel_1' in percentiles
        assert 'channel_2' in percentiles
        
        for channel_key in percentiles:
            low, high = percentiles[channel_key]
            assert low < high
            assert 0 <= low <= 255
            assert 0 <= high <= 255
    
    def test_normalize_image_file(self, sample_image, temp_reference_file):
        """Test normalization on image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample image
            input_path = Path(tmpdir) / 'input.png'
            output_path = Path(tmpdir) / 'output.png'
            cv2.imwrite(str(input_path), sample_image)
            
            # Apply normalization
            normalizer = HistogramPercentileNormalizer(temp_reference_file)
            success = normalizer.normalize_image_file(input_path, output_path)
            
            assert success
            assert output_path.exists()
            
            # Load and verify output
            normalized = cv2.imread(str(output_path))
            assert normalized is not None
            assert normalized.shape == sample_image.shape
    
    def test_normalize_uniform_image(self, temp_reference_file):
        """Test normalization on uniform (constant) image."""
        # Create uniform image
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        normalizer = HistogramPercentileNormalizer(temp_reference_file)
        normalized = normalizer.normalize_image(img)
        
        # Should handle uniform image gracefully
        assert normalized.shape == img.shape
        assert normalized.dtype == np.uint8


class TestBatchHistogramNormalization:
    """Test suite for batch histogram normalization."""
    
    @pytest.fixture
    def sample_images(self):
        """Create multiple sample images."""
        images = []
        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :, 0] = 80 + i * 20
            img[:, :, 1] = 140 + i * 10
            img[:, :, 2] = 200 - i * 15
            images.append(img)
        return images
    
    @pytest.fixture
    def temp_image_dir(self, sample_images):
        """Create temporary directory with sample images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i, img in enumerate(sample_images):
                cv2.imwrite(str(tmpdir / f'image_{i}.png'), img)
            yield tmpdir
    
    @pytest.fixture
    def temp_reference_file(self, sample_images):
        """Create a temporary reference image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, sample_images[0])
            yield f.name
        Path(f.name).unlink()
    
    def test_batch_normalization(self, temp_image_dir, temp_reference_file):
        """Test batch normalization on multiple images."""
        with tempfile.TemporaryDirectory() as output_dir:
            successful, failed = batch_histogram_normalization(
                temp_image_dir,
                output_dir,
                temp_reference_file,
                percentiles=(1.0, 99.0),
                file_pattern='*.png'
            )
            
            assert successful == 3
            assert failed == 0
            
            # Verify output files exist
            output_files = list(Path(output_dir).glob('*.png'))
            assert len(output_files) == 3
    
    def test_batch_normalization_custom_percentiles(
        self, temp_image_dir, temp_reference_file
    ):
        """Test batch normalization with custom percentiles."""
        with tempfile.TemporaryDirectory() as output_dir:
            successful, failed = batch_histogram_normalization(
                temp_image_dir,
                output_dir,
                temp_reference_file,
                percentiles=(5.0, 95.0),
                file_pattern='*.png'
            )
            
            assert successful == 3
            assert failed == 0


class TestHistogramComparison:
    """Test suite for histogram comparison utilities."""
    
    @pytest.fixture
    def image1(self):
        """Create first test image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 100
        img[:, :, 1] = 150
        img[:, :, 2] = 200
        return img
    
    @pytest.fixture
    def image2(self):
        """Create second test image with different distribution."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 120
        img[:, :, 1] = 160
        img[:, :, 2] = 180
        return img
    
    @pytest.fixture
    def identical_image(self, image1):
        """Create an identical copy of image1."""
        return image1.copy()
    
    def test_compare_histograms(self, image1, image2):
        """Test histogram comparison between two images."""
        metrics = compare_histograms(image1, image2)
        
        assert 'channel_0' in metrics
        assert 'channel_1' in metrics
        assert 'channel_2' in metrics
        
        for channel_key in metrics:
            assert 'correlation' in metrics[channel_key]
            assert 'chi_square' in metrics[channel_key]
            assert 'intersection' in metrics[channel_key]
    
    def test_identical_images_high_correlation(self, image1, identical_image):
        """Test that identical images have high correlation."""
        metrics = compare_histograms(image1, identical_image)
        
        for channel_key in metrics:
            # Correlation should be very close to 1.0 for identical images
            assert metrics[channel_key]['correlation'] > 0.99
            # Chi-square should be very close to 0 for identical images
            assert metrics[channel_key]['chi_square'] < 0.01
    
    def test_different_images_lower_correlation(self, image1, image2):
        """Test that different images have lower correlation."""
        metrics = compare_histograms(image1, image2)
        
        for channel_key in metrics:
            # Correlation should be less than 1.0 for different images
            assert metrics[channel_key]['correlation'] < 1.0
            # Chi-square should be greater than 0 for different images
            assert metrics[channel_key]['chi_square'] > 0


class TestHistogramNormalizationIntegration:
    """Integration tests for histogram percentile normalization."""
    
    def test_normalization_reduces_divergence(self):
        """Test that normalization reduces color divergence."""
        # Create reference image
        ref = np.random.randint(100, 150, (200, 200, 3), dtype=np.uint8)
        
        # Create target image with different distribution
        target = np.random.randint(50, 100, (200, 200, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, ref)
            ref_path = f.name
        
        try:
            # Compute metrics before normalization
            metrics_before = compare_histograms(ref, target)
            
            # Apply normalization
            normalizer = HistogramPercentileNormalizer(ref_path)
            normalized = normalizer.normalize_image(target)
            
            # Compute metrics after normalization
            metrics_after = compare_histograms(ref, normalized)
            
            # Check that correlation improved for at least one channel
            improved = False
            for channel in range(3):
                channel_key = f'channel_{channel}'
                if metrics_after[channel_key]['correlation'] > metrics_before[channel_key]['correlation']:
                    improved = True
                    break
            
            assert improved, "Normalization should improve correlation with reference"
            
        finally:
            Path(ref_path).unlink()
    
    def test_normalization_with_histology_colors(self):
        """Test normalization with colors typical of histological images."""
        # Simulate H&E stained tissue colors
        ref = np.zeros((200, 200, 3), dtype=np.uint8)
        ref[50:150, 50:150, 0] = 160  # Blue channel (nuclei)
        ref[50:150, 50:150, 1] = 90   # Green channel
        ref[50:150, 50:150, 2] = 210  # Red channel (cytoplasm)
        
        # Target with different staining intensity
        target = np.zeros((200, 200, 3), dtype=np.uint8)
        target[50:150, 50:150, 0] = 120
        target[50:150, 50:150, 1] = 70
        target[50:150, 50:150, 2] = 180
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, ref)
            ref_path = f.name
        
        try:
            normalizer = HistogramPercentileNormalizer(ref_path)
            normalized = normalizer.normalize_image(target)
            
            # Verify normalization was applied
            assert not np.array_equal(normalized, target)
            
            # Verify output is valid
            assert normalized.dtype == np.uint8
            assert np.all(normalized >= 0) and np.all(normalized <= 255)
            
        finally:
            Path(ref_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
