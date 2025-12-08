"""
Tests for unified color correction and normalization module.

This module tests:
- ColorProcessor with all methods (reinhard, histogram, percentile_mapping)
- Backup and restore functionality
- Batch processing with backup/restore
- Directory handling patterns
- Config integration
"""

import pytest
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from autoslide.utils.color_correction import (
    ColorProcessor,
    batch_process_directory,
    batch_process_suggested_regions,
    find_svs_image_directories,
    restore_originals,
    list_backups
)


class TestColorProcessor:
    """Test suite for unified ColorProcessor class."""
    
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
    
    def test_processor_reinhard_method(self, sample_image, temp_reference_file):
        """Test ColorProcessor with Reinhard method."""
        processor = ColorProcessor(temp_reference_file, method='reinhard')
        processed = processor.process_image(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == np.uint8
        assert not np.array_equal(processed, sample_image)
    
    def test_processor_histogram_method(self, sample_image, temp_reference_file):
        """Test ColorProcessor with histogram method."""
        processor = ColorProcessor(temp_reference_file, method='histogram')
        processed = processor.process_image(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == np.uint8
    
    def test_processor_percentile_mapping_method(self, sample_image, temp_reference_file):
        """Test ColorProcessor with percentile_mapping method."""
        processor = ColorProcessor(
            temp_reference_file,
            method='percentile_mapping',
            percentiles=np.linspace(0, 100, 101)
        )
        processed = processor.process_image(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == np.uint8
        assert not np.array_equal(processed, sample_image)
    
    def test_processor_invalid_method(self, temp_reference_file):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            ColorProcessor(temp_reference_file, method='invalid_method')
    
    def test_processor_multiple_references(self, temp_reference_file):
        """Test ColorProcessor with multiple reference images."""
        processor = ColorProcessor(
            [temp_reference_file, temp_reference_file],
            method='reinhard'
        )
        assert len(processor.reference_images) == 2
    
    def test_process_image_file(self, sample_image, temp_reference_file):
        """Test processing an image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'input.png'
            output_path = Path(tmpdir) / 'output.png'
            cv2.imwrite(str(input_path), sample_image)
            
            processor = ColorProcessor(temp_reference_file, method='reinhard')
            success = processor.process_image_file(input_path, output_path)
            
            assert success
            assert output_path.exists()
            
            processed = cv2.imread(str(output_path))
            assert processed is not None
            assert processed.shape == sample_image.shape


class TestBackupRestore:
    """Test suite for backup and restore functionality."""
    
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
    
    def test_backup_images(self, temp_image_dir):
        """Test backing up images."""
        from autoslide.utils.color_correction.color_processor import backup_images
        
        with tempfile.TemporaryDirectory() as backup_dir:
            successful, failed = backup_images(
                temp_image_dir,
                backup_dir,
                file_pattern='*.png'
            )
            
            assert successful == 3
            assert failed == 0
            
            # Verify backup files exist
            backup_files = list(Path(backup_dir).glob('*.png'))
            assert len(backup_files) == 3
            
            # Verify metadata exists
            metadata_path = Path(backup_dir) / 'backup_metadata.json'
            assert metadata_path.exists()
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert 'backup_timestamp' in metadata
            assert 'source_directory' in metadata
            assert 'file_pattern' in metadata
    
    def test_restore_originals(self, temp_image_dir):
        """Test restoring original images from backup."""
        from autoslide.utils.color_correction.color_processor import backup_images
        
        with tempfile.TemporaryDirectory() as backup_dir:
            # Create backup
            backup_images(temp_image_dir, backup_dir, file_pattern='*.png')
            
            # Modify original images
            for img_path in Path(temp_image_dir).glob('*.png'):
                img = cv2.imread(str(img_path))
                img[:, :, :] = 0  # Make all black
                cv2.imwrite(str(img_path), img)
            
            # Restore from backup
            successful, failed = restore_originals(backup_dir, temp_image_dir)
            
            assert successful == 3
            assert failed == 0
            
            # Verify images were restored
            for img_path in Path(temp_image_dir).glob('*.png'):
                img = cv2.imread(str(img_path))
                # Should not be all black anymore
                assert not np.all(img == 0)
    
    def test_list_backups(self, temp_image_dir):
        """Test listing available backups."""
        from autoslide.utils.color_correction.color_processor import backup_images
        
        with tempfile.TemporaryDirectory() as backup_root:
            backup_root = Path(backup_root)
            
            # Create multiple backups
            for i in range(2):
                backup_dir = backup_root / f'backup_{i}'
                backup_images(
                    temp_image_dir,
                    backup_dir,
                    file_pattern='*.png',
                    metadata={'method': 'reinhard'}
                )
            
            # List backups
            backups = list_backups(backup_root)
            
            assert len(backups) == 2
            for backup in backups:
                assert 'backup_dir' in backup
                assert 'timestamp' in backup
                assert 'source_directory' in backup
                assert 'file_count' in backup
                assert backup['file_count'] == 3


class TestBatchProcessing:
    """Test suite for batch processing with backup/restore."""
    
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
    
    def test_batch_process_with_backup(self, temp_image_dir, temp_reference_file):
        """Test batch processing with backup enabled."""
        result = batch_process_directory(
            input_dir=temp_image_dir,
            reference_images=temp_reference_file,
            method='reinhard',
            backup=True,
            replace_originals=True
        )
        
        assert result['success']
        assert result['processed'] == 3
        assert result['failed'] == 0
        assert 'backup_dir' in result
        assert result['backup_count'] == 3
        
        # Verify backup exists
        backup_dir = Path(result['backup_dir'])
        assert backup_dir.exists()
        assert len(list(backup_dir.glob('*.png'))) == 3
    
    def test_batch_process_without_backup(self, temp_image_dir, temp_reference_file):
        """Test batch processing without backup."""
        result = batch_process_directory(
            input_dir=temp_image_dir,
            reference_images=temp_reference_file,
            method='reinhard',
            backup=False,
            replace_originals=True
        )
        
        assert result['success']
        assert result['processed'] == 3
        assert result['failed'] == 0
        assert 'backup_dir' not in result
    
    def test_batch_process_to_output_dir(self, temp_image_dir, temp_reference_file):
        """Test batch processing to separate output directory."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = batch_process_directory(
                input_dir=temp_image_dir,
                reference_images=temp_reference_file,
                method='reinhard',
                backup=False,
                replace_originals=False,
                output_dir=output_dir
            )
            
            assert result['success']
            assert result['processed'] == 3
            assert result['output_dir'] == str(output_dir)
            
            # Verify output files exist
            output_files = list(Path(output_dir).glob('*.png'))
            assert len(output_files) == 3
    
    def test_batch_process_percentile_mapping_method(self, temp_image_dir, temp_reference_file):
        """Test batch processing with percentile_mapping method."""
        result = batch_process_directory(
            input_dir=temp_image_dir,
            reference_images=temp_reference_file,
            method='percentile_mapping',
            percentiles=np.linspace(0, 100, 101),
            backup=True,
            replace_originals=True
        )
        
        assert result['success']
        assert result['processed'] == 3
        assert result['method'] == 'percentile_mapping'


class TestDirectoryHandling:
    """Test suite for directory structure handling."""
    
    def test_nested_directory_structure(self):
        """Test handling of nested directory structures like in prediction.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create nested structure: suggested_regions/SVS_NAME/images/*.png
            svs_dir = tmpdir / 'suggested_regions' / 'SVS_001' / 'images'
            svs_dir.mkdir(parents=True)
            
            # Create sample images
            for i in range(2):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                img[:, :, :] = 100 + i * 50
                cv2.imwrite(str(svs_dir / f'image_{i}.png'), img)
            
            # Create reference
            ref_img = np.zeros((100, 100, 3), dtype=np.uint8)
            ref_img[:, :, :] = 150
            ref_path = tmpdir / 'reference.png'
            cv2.imwrite(str(ref_path), ref_img)
            
            # Process with backup
            result = batch_process_directory(
                input_dir=svs_dir,
                reference_images=ref_path,
                method='reinhard',
                backup=True,
                replace_originals=True
            )
            
            assert result['success']
            assert result['processed'] == 2
            
            # Verify backup was created
            backup_dir = Path(result['backup_dir'])
            assert backup_dir.exists()
    
    def test_restore_to_different_location(self):
        """Test restoring backups to a different location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create original images
            original_dir = tmpdir / 'original'
            original_dir.mkdir()
            
            for i in range(2):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                img[:, :, :] = 100 + i * 50
                cv2.imwrite(str(original_dir / f'image_{i}.png'), img)
            
            # Create backup
            from autoslide.utils.color_correction.color_processor import backup_images
            backup_dir = tmpdir / 'backup'
            backup_images(original_dir, backup_dir, file_pattern='*.png')
            
            # Restore to different location
            restore_dir = tmpdir / 'restored'
            restore_dir.mkdir()
            
            successful, failed = restore_originals(backup_dir, restore_dir)
            
            assert successful == 2
            assert failed == 0
            
            # Verify files in restore location
            restored_files = list(restore_dir.glob('*.png'))
            assert len(restored_files) == 2


class TestConfigIntegration:
    """Test suite for autoslide config integration."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock autoslide config."""
        return {
            'data_dir': '/tmp/test_data',
            'suggested_regions_dir': '/tmp/test_data/suggested_regions'
        }
    
    @pytest.fixture
    def sample_images(self):
        """Create multiple sample images."""
        images = []
        for i in range(2):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :, 0] = 100 + i * 10
            img[:, :, 1] = 150 + i * 10
            img[:, :, 2] = 200 - i * 10
            images.append(img)
        return images
    
    def test_find_svs_image_directories(self, sample_images):
        """Test finding SVS image directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create suggested_regions structure
            suggested_regions = tmpdir / 'suggested_regions'
            
            # Create multiple SVS directories
            for svs_name in ['SVS_001', 'SVS_002']:
                images_dir = suggested_regions / svs_name / 'images'
                images_dir.mkdir(parents=True)
                
                # Add sample images
                for i, img in enumerate(sample_images):
                    cv2.imwrite(str(images_dir / f'image_{i}.png'), img)
            
            # Find all directories
            image_dirs = find_svs_image_directories(suggested_regions)
            
            assert len(image_dirs) == 2
            assert all(d.name == 'images' for d in image_dirs)
    
    def test_find_specific_svs(self, sample_images):
        """Test finding specific SVS directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            suggested_regions = tmpdir / 'suggested_regions'
            
            # Create multiple SVS directories
            for svs_name in ['SVS_001', 'SVS_002']:
                images_dir = suggested_regions / svs_name / 'images'
                images_dir.mkdir(parents=True)
                
                for i, img in enumerate(sample_images):
                    cv2.imwrite(str(images_dir / f'image_{i}.png'), img)
            
            # Find specific SVS
            image_dirs = find_svs_image_directories(suggested_regions, svs_name='SVS_001')
            
            assert len(image_dirs) == 1
            assert image_dirs[0].parent.name == 'SVS_001'
    
    def test_batch_process_suggested_regions(self, sample_images):
        """Test batch processing of suggested_regions structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            suggested_regions = tmpdir / 'suggested_regions'
            
            # Create SVS directories
            for svs_name in ['SVS_001', 'SVS_002']:
                images_dir = suggested_regions / svs_name / 'images'
                images_dir.mkdir(parents=True)
                
                for i, img in enumerate(sample_images):
                    cv2.imwrite(str(images_dir / f'image_{i}.png'), img)
            
            # Create reference image
            ref_path = tmpdir / 'reference.png'
            cv2.imwrite(str(ref_path), sample_images[0])
            
            # Process all SVS directories
            result = batch_process_suggested_regions(
                reference_images=ref_path,
                method='reinhard',
                backup=True,
                replace_originals=True,
                suggested_regions_dir=suggested_regions
            )
            
            assert result['success']
            assert result['svs_count'] == 2
            assert result['total_processed'] == 4  # 2 images per SVS
            assert result['total_failed'] == 0
            assert 'SVS_001' in result['svs_results']
            assert 'SVS_002' in result['svs_results']
            
            # Verify backups were created
            for svs_name in ['SVS_001', 'SVS_002']:
                backup_root = suggested_regions / svs_name / 'images' / 'backups'
                assert backup_root.exists()
    
    def test_batch_process_specific_svs(self, sample_images):
        """Test processing specific SVS only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            suggested_regions = tmpdir / 'suggested_regions'
            
            # Create SVS directories
            for svs_name in ['SVS_001', 'SVS_002']:
                images_dir = suggested_regions / svs_name / 'images'
                images_dir.mkdir(parents=True)
                
                for i, img in enumerate(sample_images):
                    cv2.imwrite(str(images_dir / f'image_{i}.png'), img)
            
            # Create reference image
            ref_path = tmpdir / 'reference.png'
            cv2.imwrite(str(ref_path), sample_images[0])
            
            # Process only SVS_001
            result = batch_process_suggested_regions(
                reference_images=ref_path,
                method='reinhard',
                backup=False,
                replace_originals=True,
                svs_name='SVS_001',
                suggested_regions_dir=suggested_regions
            )
            
            assert result['success']
            assert result['svs_count'] == 1
            assert result['total_processed'] == 2
            assert 'SVS_001' in result['svs_results']
            assert 'SVS_002' not in result['svs_results']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
