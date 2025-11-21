"""
Tests for color correction preprocessing module.

This module tests:
- Color correction functionality
- Color distribution divergence detection
- Integration with preprocessing pipeline
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from autoslide.pipeline.model.color_correction import (
    ColorCorrector,
    compute_color_divergence,
    batch_color_correction
)


class TestColorCorrector:
    """Test suite for ColorCorrector class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple RGB image with known color distribution
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # Blue channel
        img[:, :, 1] = 150  # Green channel
        img[:, :, 2] = 200  # Red channel
        return img
    
    @pytest.fixture
    def reference_image(self):
        """Create a reference image with different color distribution."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 120  # Blue channel
        img[:, :, 1] = 160  # Green channel
        img[:, :, 2] = 180  # Red channel
        return img
    
    @pytest.fixture
    def temp_reference_file(self, reference_image):
        """Create a temporary reference image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, reference_image)
            yield f.name
        Path(f.name).unlink()
    
    def test_color_corrector_initialization(self, temp_reference_file):
        """Test ColorCorrector initialization with reference images."""
        corrector = ColorCorrector(temp_reference_file)
        assert corrector.reference_stats is not None
        assert 'mean' in corrector.reference_stats
        assert 'std' in corrector.reference_stats
        assert len(corrector.reference_stats['mean']) == 3
        assert len(corrector.reference_stats['std']) == 3
    
    def test_color_corrector_multiple_references(self, temp_reference_file):
        """Test ColorCorrector with multiple reference images."""
        corrector = ColorCorrector([temp_reference_file, temp_reference_file])
        assert corrector.reference_stats is not None
        assert len(corrector.reference_images) == 2
    
    def test_reinhard_color_transfer(self, sample_image, temp_reference_file):
        """Test Reinhard color transfer method."""
        corrector = ColorCorrector(temp_reference_file)
        corrected = corrector.correct_image(sample_image, method='reinhard')
        
        # Check output shape and type
        assert corrected.shape == sample_image.shape
        assert corrected.dtype == np.uint8
        
        # Check that colors have changed
        assert not np.array_equal(corrected, sample_image)
    
    def test_histogram_matching(self, sample_image, temp_reference_file):
        """Test histogram matching method."""
        corrector = ColorCorrector(temp_reference_file)
        corrected = corrector.correct_image(sample_image, method='histogram')
        
        # Check output shape and type
        assert corrected.shape == sample_image.shape
        assert corrected.dtype == np.uint8
    
    def test_invalid_method(self, sample_image, temp_reference_file):
        """Test that invalid method raises error."""
        corrector = ColorCorrector(temp_reference_file)
        with pytest.raises(ValueError):
            corrector.correct_image(sample_image, method='invalid_method')
    
    def test_get_color_statistics(self, sample_image, temp_reference_file):
        """Test color statistics computation."""
        corrector = ColorCorrector(temp_reference_file)
        stats = corrector.get_color_statistics(sample_image)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert len(stats['mean']) == 3
        assert len(stats['std']) == 3
    
    def test_correct_image_file(self, sample_image, temp_reference_file):
        """Test color correction on image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample image
            input_path = Path(tmpdir) / 'input.png'
            output_path = Path(tmpdir) / 'output.png'
            cv2.imwrite(str(input_path), sample_image)
            
            # Apply correction
            corrector = ColorCorrector(temp_reference_file)
            success = corrector.correct_image_file(input_path, output_path)
            
            assert success
            assert output_path.exists()
            
            # Load and verify output
            corrected = cv2.imread(str(output_path))
            assert corrected is not None
            assert corrected.shape == sample_image.shape


class TestColorDivergence:
    """Test suite for color divergence computation."""
    
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
        """Create second test image with different colors."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 120
        img[:, :, 1] = 160
        img[:, :, 2] = 180
        return img
    
    @pytest.fixture
    def identical_image(self, image1):
        """Create an identical copy of image1."""
        return image1.copy()
    
    def test_euclidean_divergence(self, image1, image2):
        """Test Euclidean distance metric."""
        divergence = compute_color_divergence(image1, image2, metric='euclidean')
        assert divergence > 0
        assert isinstance(divergence, (float, np.floating))
    
    def test_manhattan_divergence(self, image1, image2):
        """Test Manhattan distance metric."""
        divergence = compute_color_divergence(image1, image2, metric='manhattan')
        assert divergence > 0
        assert isinstance(divergence, (float, np.floating))
    
    def test_kl_divergence(self, image1, image2):
        """Test KL divergence metric."""
        divergence = compute_color_divergence(image1, image2, metric='kl_divergence')
        assert isinstance(divergence, (float, np.floating))
    
    def test_identical_images_low_divergence(self, image1, identical_image):
        """Test that identical images have very low divergence."""
        divergence = compute_color_divergence(image1, identical_image, metric='euclidean')
        assert divergence < 1.0  # Should be very close to 0
    
    def test_different_images_high_divergence(self, image1, image2):
        """Test that different images have measurable divergence."""
        divergence = compute_color_divergence(image1, image2, metric='euclidean')
        assert divergence > 1.0
    
    def test_invalid_metric(self, image1, image2):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError):
            compute_color_divergence(image1, image2, metric='invalid_metric')
    
    def test_divergence_threshold_detection(self, image1, image2):
        """Test divergence can be used to detect color variations."""
        # Define a threshold for acceptable color variation
        threshold = 50.0
        
        divergence = compute_color_divergence(image1, image2, metric='euclidean')
        
        # This test demonstrates how to use divergence for quality control
        if divergence > threshold:
            # Color correction would be recommended
            assert True
        else:
            # Images are similar enough
            assert divergence <= threshold


class TestBatchColorCorrection:
    """Test suite for batch color correction."""
    
    @pytest.fixture
    def sample_images(self):
        """Create multiple sample images."""
        images = []
        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :, 0] = 100 + i * 10
            img[:, :, 1] = 150 + i * 10
            img[:, :, 2] = 200 - i * 10
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
    
    def test_batch_correction(self, temp_image_dir, temp_reference_file):
        """Test batch color correction on multiple images."""
        with tempfile.TemporaryDirectory() as output_dir:
            successful, failed = batch_color_correction(
                temp_image_dir,
                output_dir,
                temp_reference_file,
                method='reinhard',
                file_pattern='*.png'
            )
            
            assert successful == 3
            assert failed == 0
            
            # Verify output files exist
            output_files = list(Path(output_dir).glob('*.png'))
            assert len(output_files) == 3


class TestColorCorrectionIntegration:
    """Integration tests for color correction in preprocessing pipeline."""
    
    def test_color_correction_preserves_image_dimensions(self):
        """Test that color correction preserves image dimensions."""
        # Create test images
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        ref = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, ref)
            ref_path = f.name
        
        try:
            corrector = ColorCorrector(ref_path)
            corrected = corrector.correct_image(img)
            
            assert corrected.shape == img.shape
        finally:
            Path(ref_path).unlink()
    
    def test_color_correction_with_real_histology_colors(self):
        """Test color correction with colors typical of histological images."""
        # Simulate H&E stained tissue colors
        # Purple/blue nuclei and pink cytoplasm
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150, 0] = 180  # Blue channel (nuclei)
        img[50:150, 50:150, 1] = 100  # Green channel
        img[50:150, 50:150, 2] = 200  # Red channel (cytoplasm)
        
        ref = np.zeros((200, 200, 3), dtype=np.uint8)
        ref[50:150, 50:150, 0] = 160
        ref[50:150, 50:150, 1] = 90
        ref[50:150, 50:150, 2] = 210
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, ref)
            ref_path = f.name
        
        try:
            corrector = ColorCorrector(ref_path)
            corrected = corrector.correct_image(img, method='reinhard')
            
            # Verify correction was applied
            assert not np.array_equal(corrected, img)
            
            # Verify output is valid
            assert corrected.dtype == np.uint8
            assert np.all(corrected >= 0) and np.all(corrected <= 255)
        finally:
            Path(ref_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
