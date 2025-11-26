# Color Correction and Normalization Module

Unified module for color correction and normalization of histological images with backup/restore functionality.

## Features

- **Multiple Methods**: Supports Reinhard color transfer, histogram matching, and percentile mapping
- **Backup/Restore**: Automatically backup original images before processing and restore when needed
- **Batch Processing**: Process entire directories with a single command
- **Directory Structure Aware**: Handles nested directory structures like those in prediction.py

## Methods

### 1. Reinhard Color Transfer (`reinhard`)
Based on "Color Transfer between Images" by Reinhard et al., 2001. Transfers color characteristics in LAB color space by matching mean and standard deviation.

**Best for**: General color correction when reference images have the desired color characteristics.

### 2. Histogram Matching (`histogram`)
Matches histogram statistics in LAB color space for each channel independently.

**Best for**: When you want to match overall intensity distributions.

### 3. Percentile Mapping (`percentile_mapping`)
Maps intensity values using percentile-based interpolation. Computes percentile values (0-100) for each channel in both source and target images, then uses interpolation to transform source intensities to match target distribution.

**Best for**: Normalizing staining intensity variations across different batches or scanning sessions. Particularly effective for histological images where overall intensity distribution matters more than absolute color values.

**Implementation**: Based on the method from `autoslide_analysis/src/color_corrections_test.py`, using 101 percentile points (0, 1, 2, ..., 100) for smooth interpolation.

## Usage

### Command Line Interface

#### Process all suggested_regions (uses autoslide config)
```bash
# Process all SVS directories in suggested_regions
python -m autoslide.utils.color_correction process-pipeline \
    --reference-images data/reference/ref.png \
    --method reinhard \
    --backup
```

#### Process specific SVS directory
```bash
# Process only SVS_001
python -m autoslide.utils.color_correction process-pipeline \
    --reference-images data/reference/ref.png \
    --svs-name SVS_001 \
    --method reinhard \
    --backup
```

#### Process with percentile mapping (pipeline)
```bash
python -m autoslide.utils.color_correction process-pipeline \
    --reference-images "data/reference/*.png" \
    --method percentile_mapping \
    --backup
```

#### Process arbitrary directory (not using config)
```bash
python -m autoslide.utils.color_correction process \
    --input-dir data/images \
    --reference-images data/reference/ref.png \
    --method reinhard \
    --backup
```

#### Save to output directory instead of replacing originals
```bash
python -m autoslide.utils.color_correction process \
    --input-dir data/images \
    --reference-images data/reference/ref.png \
    --method reinhard \
    --no-replace \
    --output-dir data/images_corrected
```

#### Restore original images from backup
```bash
python -m autoslide.utils.color_correction restore \
    --backup-dir data/images/backups/backup_20231126_120000
```

#### List available backups
```bash
python -m autoslide.utils.color_correction list-backups \
    --backup-root data/images/backups
```

### Python API

#### Process suggested_regions (uses autoslide config)
```python
from autoslide.utils.color_correction import batch_process_suggested_regions

# Process all SVS directories
result = batch_process_suggested_regions(
    reference_images='data/reference/ref.png',
    method='reinhard',
    backup=True,
    replace_originals=True
)

print(f"Processed {result['svs_count']} SVS directories")
print(f"Total images: {result['total_processed']}")

# Process specific SVS
result = batch_process_suggested_regions(
    reference_images='data/reference/ref.png',
    method='reinhard',
    backup=True,
    svs_name='SVS_001'
)
```

#### Find SVS image directories
```python
from autoslide.utils.color_correction import find_svs_image_directories

# Find all SVS image directories
image_dirs = find_svs_image_directories()

# Find specific SVS
image_dirs = find_svs_image_directories(svs_name='SVS_001')
```

#### Basic usage
```python
from autoslide.utils.color_correction import ColorProcessor

# Initialize processor with reference images
processor = ColorProcessor(
    reference_images='path/to/reference.png',
    method='reinhard'
)

# Process a single image
import cv2
image = cv2.imread('path/to/image.png')
processed = processor.process_image(image)
cv2.imwrite('path/to/output.png', processed)
```

#### Batch processing arbitrary directory
```python
from autoslide.utils.color_correction import batch_process_directory

result = batch_process_directory(
    input_dir='data/images',
    reference_images='data/reference/ref.png',
    method='reinhard',
    backup=True,
    replace_originals=True
)

print(f"Processed: {result['processed']} images")
print(f"Backup location: {result['backup_dir']}")
```

#### Restore from backup
```python
from autoslide.utils.color_correction import restore_originals

successful, failed = restore_originals(
    backup_dir='data/images/backups/backup_20231126_120000'
)

print(f"Restored {successful} images")
```

#### List backups
```python
from autoslide.utils.color_correction import list_backups

backups = list_backups('data/images/backups')
for backup in backups:
    print(f"Backup: {backup['timestamp']}")
    print(f"  Files: {backup['file_count']}")
    print(f"  Method: {backup['method']}")
```

## Workflow Example

### Pipeline Workflow (Recommended)

#### 1. Process all suggested_regions with backup
```bash
python -m autoslide.utils.color_correction process-pipeline \
    --reference-images data/reference/*.png \
    --method reinhard \
    --backup
```

This will:
- Process all SVS directories in `suggested_regions/` (from config)
- Create backups in `suggested_regions/SVS_NAME/images/backups/backup_TIMESTAMP/`
- Replace original images with color-corrected versions
- Store metadata about the processing

#### 2. If results are not satisfactory, restore originals
```bash
# Find available backups
python -m autoslide.utils.color_correction list-backups \
    --backup-root data/suggested_regions/SVS_001/images/backups

# Restore from specific backup
python -m autoslide.utils.color_correction restore \
    --backup-dir data/suggested_regions/SVS_001/images/backups/backup_TIMESTAMP
```

#### 3. Try different method
```bash
python -m autoslide.utils.color_correction process-pipeline \
    --reference-images data/reference/*.png \
    --method percentile_mapping \
    --backup
```

### Manual Workflow (Specific Directory)

#### 1. Process specific directory with backup
```bash
python -m autoslide.utils.color_correction process \
    --input-dir data/suggested_regions/SVS_001/images \
    --reference-images data/reference/*.png \
    --method reinhard \
    --backup
```

This will:
- Create a backup in `data/suggested_regions/SVS_001/images/backups/backup_TIMESTAMP/`
- Replace original images with color-corrected versions
- Store metadata about the processing

#### 2. Restore if needed
```bash
python -m autoslide.utils.color_correction restore \
    --backup-dir data/suggested_regions/SVS_001/images/backups/backup_TIMESTAMP
```

#### 3. Try different method
```bash
python -m autoslide.utils.color_correction process \
    --input-dir data/suggested_regions/SVS_001/images \
    --reference-images data/reference/*.png \
    --method percentile_mapping \
    --backup
```

## Directory Structure

The module handles directory structures similar to those in `model/prediction.py` and `calc_fibrosis.py`:

```
data/
├── suggested_regions/
│   ├── SVS_001/
│   │   ├── images/
│   │   │   ├── image_001.png
│   │   │   ├── image_002.png
│   │   │   └── backups/
│   │   │       └── backup_20231126_120000/
│   │   │           ├── image_001.png
│   │   │           ├── image_002.png
│   │   │           └── backup_metadata.json
│   │   ├── masks/
│   │   └── overlays/
│   └── SVS_002/
│       └── images/
└── reference/
    └── ref.png
```

## Backup Metadata

Each backup includes a `backup_metadata.json` file with:
- `backup_timestamp`: When the backup was created
- `source_directory`: Original location of images
- `file_pattern`: Pattern used to select files
- `method`: Processing method used
- `replace_originals`: Whether originals were replaced

## Integration with Existing Pipeline

This module integrates with the autoslide config and can be used before prediction:

```python
from autoslide.utils.color_correction import batch_process_suggested_regions
from autoslide.pipeline.model.prediction import find_images_to_process

# 1. Apply color correction to all suggested_regions
result = batch_process_suggested_regions(
    reference_images='data/reference/*.png',
    method='reinhard',
    backup=True,
    replace_originals=True
)

print(f"Color corrected {result['total_processed']} images across {result['svs_count']} SVS directories")

# 2. Continue with prediction pipeline
images_by_svs = find_images_to_process()
# ... existing prediction code ...
```

Or process specific SVS directories:

```python
from autoslide.utils.color_correction import batch_process_directory
from autoslide.pipeline.model.prediction import find_images_to_process
from pathlib import Path

# 1. Find images to process
images_by_svs = find_images_to_process()

# 2. Apply color correction to each SVS directory
for svs_name, images in images_by_svs.items():
    if images:
        image_dir = Path(images[0]['image_path']).parent
        
        result = batch_process_directory(
            input_dir=image_dir,
            reference_images='data/reference/*.png',
            method='reinhard',
            backup=True,
            replace_originals=True
        )
        
        print(f"Processed {svs_name}: {result['processed']} images")

# 3. Continue with prediction pipeline
# ... existing prediction code ...
```

## Testing

Run tests with:
```bash
pytest tests/test_color_processor.py -v
```

## Migration from Old Modules

The new unified module replaces:
- `autoslide/utils/color_correction/color_correction.py` (Reinhard method)
- `autoslide/utils/color_correction/color_normalization.py` (Percentile method)

Old code can be updated:

**Before:**
```python
from autoslide.utils.color_correction.color_correction import ColorCorrector
corrector = ColorCorrector(reference_images)
corrected = corrector.correct_image(image, method='reinhard')
```

**After:**
```python
from autoslide.utils.color_correction import ColorProcessor
processor = ColorProcessor(reference_images, method='reinhard')
corrected = processor.process_image(image)
```

**Before:**
```python
from autoslide.utils.color_correction.color_normalization import HistogramPercentileNormalizer
normalizer = HistogramPercentileNormalizer(reference_images, percentiles=(1.0, 99.0))
normalized = normalizer.normalize_image(image)
```

**After:**
```python
from autoslide.utils.color_correction import ColorProcessor
import numpy as np
processor = ColorProcessor(reference_images, method='percentile_mapping', percentiles=np.linspace(0, 100, 101))
normalized = processor.process_image(image)
```
