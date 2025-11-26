# Color Correction Preprocessing

## Overview

This module implements color correction preprocessing for histological images to normalize color variations across different datasets or scanning sessions. This addresses issue #80 by providing project-level color correction using reference images from the original dataset.

## Features

- **Multiple Color Transfer Methods**: Supports Reinhard color transfer and histogram matching
- **Reference-Based Correction**: Uses example images from the original dataset to correct new datasets
- **Batch Processing**: Efficiently process entire directories of images
- **Color Divergence Detection**: Test for color distribution differences to determine if correction is needed
- **Integration Ready**: Seamlessly integrates with existing preprocessing pipeline

## Installation

No additional dependencies required! The module uses existing dependencies:
- `numpy >= 1.20.0`
- `opencv-python >= 4.5.0`
- `scipy >= 1.7.0`

## Usage

### Basic Usage

```python
from autoslide.pipeline.model.color_correction import ColorCorrector
import cv2

# Initialize with reference image(s) from original dataset
corrector = ColorCorrector('path/to/reference_image.png')

# Load and correct an image
img = cv2.imread('new_dataset/image.png')
corrected = corrector.correct_image(img, method='reinhard')
cv2.imwrite('corrected_image.png', corrected)
```

### Using Multiple Reference Images

```python
# Use multiple reference images for more robust statistics
corrector = ColorCorrector([
    'original_dataset/sample1.png',
    'original_dataset/sample2.png',
    'original_dataset/sample3.png'
])

corrected = corrector.correct_image(img, method='reinhard')
```

### Batch Processing

```python
from autoslide.pipeline.model.data_preprocessing import apply_color_correction

# Correct all images in a directory
successful, failed = apply_color_correction(
    img_dir='data/new_dataset/images',
    output_dir='data/new_dataset/corrected',
    reference_images=[
        'data/original_dataset/ref1.png',
        'data/original_dataset/ref2.png'
    ],
    method='reinhard',
    file_pattern='*.png'
)

print(f"Corrected {successful} images, {failed} failed")
```

### Detecting Color Divergence

```python
from autoslide.pipeline.model.color_correction import compute_color_divergence
import cv2

# Load images from different datasets
ref_img = cv2.imread('original_dataset/reference.png')
new_img = cv2.imread('new_dataset/sample.png')

# Compute divergence
divergence = compute_color_divergence(ref_img, new_img, metric='euclidean')
print(f'Color divergence: {divergence}')

# Define threshold based on your requirements
DIVERGENCE_THRESHOLD = 50.0

if divergence > DIVERGENCE_THRESHOLD:
    print('⚠️  High color divergence detected - color correction recommended')
else:
    print('✅ Color distributions are similar - no correction needed')
```

## Color Correction Methods

### Reinhard Color Transfer (Recommended)

Based on "Color Transfer between Images" by Reinhard et al., 2001. This method:
- Converts images to LAB color space
- Matches mean and standard deviation of each channel
- Preserves image structure while adjusting colors
- Works well for histological images

```python
corrected = corrector.correct_image(img, method='reinhard')
```

### Histogram Matching

Matches the histogram distribution of the target image to the reference:
- Operates in LAB color space
- Adjusts each channel independently
- Good for images with similar content

```python
corrected = corrector.correct_image(img, method='histogram')
```

## Divergence Metrics

### Euclidean Distance (Default)

Computes Euclidean distance between color statistics:
```python
divergence = compute_color_divergence(img1, img2, metric='euclidean')
```

### Manhattan Distance

Computes Manhattan (L1) distance:
```python
divergence = compute_color_divergence(img1, img2, metric='manhattan')
```

### KL Divergence

Approximates KL divergence assuming Gaussian distributions:
```python
divergence = compute_color_divergence(img1, img2, metric='kl_divergence')
```

## Integration with Training Pipeline

### Before Training

```python
from autoslide.pipeline.model.data_preprocessing import (
    apply_color_correction,
    load_data,
    split_train_val
)

# Step 1: Apply color correction to new dataset
print("Applying color correction...")
apply_color_correction(
    img_dir='data/new_dataset/images',
    output_dir='data/new_dataset/corrected',
    reference_images=[
        'data/original_dataset/sample1.png',
        'data/original_dataset/sample2.png'
    ],
    method='reinhard'
)

# Step 2: Load corrected data for training
labelled_data_dir, img_dir, mask_dir, image_names, mask_names = load_data()

# Step 3: Continue with normal training pipeline
train_imgs, train_masks, val_imgs, val_masks = split_train_val(
    image_names, mask_names
)
# ... rest of training code ...
```

### Quality Control Check

```python
from autoslide.pipeline.model.color_correction import compute_color_divergence
import cv2
import os

def check_dataset_color_consistency(dataset_dir, reference_img_path, threshold=50.0):
    """Check if a dataset needs color correction."""
    ref_img = cv2.imread(reference_img_path)
    
    divergences = []
    for img_name in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            div = compute_color_divergence(ref_img, img, metric='euclidean')
            divergences.append(div)
    
    avg_divergence = sum(divergences) / len(divergences)
    max_divergence = max(divergences)
    
    print(f"Average divergence: {avg_divergence:.2f}")
    print(f"Maximum divergence: {max_divergence:.2f}")
    
    if avg_divergence > threshold:
        print("⚠️  Color correction recommended")
        return True
    else:
        print("✅ Dataset color is consistent")
        return False

# Usage
needs_correction = check_dataset_color_consistency(
    'data/new_dataset/images',
    'data/original_dataset/reference.png'
)
```

## Testing

The module includes comprehensive tests in `tests/test_color_correction.py`:

```bash
# Run tests (requires pytest and dependencies installed)
pytest tests/test_color_correction.py -v
```

Test coverage includes:
- Color corrector initialization
- Reinhard color transfer
- Histogram matching
- Color divergence computation
- Batch processing
- Integration scenarios

## API Reference

### ColorCorrector Class

```python
class ColorCorrector:
    def __init__(self, reference_images: Union[str, Path, List[Union[str, Path]]])
    def correct_image(self, image: np.ndarray, method: str = 'reinhard') -> np.ndarray
    def correct_image_file(self, input_path: Union[str, Path], 
                          output_path: Union[str, Path], 
                          method: str = 'reinhard') -> bool
    def get_color_statistics(self, image: np.ndarray) -> dict
```

### Functions

```python
def compute_color_divergence(
    image1: np.ndarray,
    image2: np.ndarray,
    metric: str = 'euclidean'
) -> float

def batch_color_correction(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reference_images: Union[str, Path, List[Union[str, Path]]],
    method: str = 'reinhard',
    file_pattern: str = '*.png'
) -> Tuple[int, int]

def apply_color_correction(
    img_dir: str,
    output_dir: str,
    reference_images: Union[str, List[str]],
    method: str = 'reinhard',
    file_pattern: str = '*.png'
) -> Tuple[int, int]
```

## Technical Details

### Color Space

All color correction operations are performed in LAB color space because:
- LAB is perceptually uniform
- Separates luminance (L) from color (A, B)
- Better for color manipulation than RGB
- Standard in color science applications

### Algorithm: Reinhard Color Transfer

1. Convert images to LAB color space
2. Compute mean (μ) and standard deviation (σ) for each channel
3. For target image:
   - Subtract target mean: `I' = I - μ_target`
   - Scale by std ratio: `I'' = I' × (σ_ref / σ_target)`
   - Add reference mean: `I''' = I'' + μ_ref`
4. Convert back to BGR color space

### Performance Considerations

- Processing time: ~10-50ms per 512×512 image (CPU)
- Memory usage: ~3× image size (for color space conversions)
- Batch processing recommended for large datasets
- Can be parallelized for multiple images

## Troubleshooting

### Issue: Colors look oversaturated

Try using histogram matching instead:
```python
corrected = corrector.correct_image(img, method='histogram')
```

### Issue: Reference images have different characteristics

Use multiple reference images to get more robust statistics:
```python
corrector = ColorCorrector([
    'ref1.png', 'ref2.png', 'ref3.png', 'ref4.png'
])
```

### Issue: Correction too aggressive

The Reinhard method preserves relative color relationships. If results are too different:
1. Check that reference images are representative
2. Verify input images are in correct format (BGR)
3. Consider using images from the same staining batch as references

## References

1. Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. IEEE Computer Graphics and Applications, 21(5), 34-41.

2. ColorTransferLib: https://github.com/hpotechius/ColorTransferLib
   - Comprehensive library for color transfer algorithms
   - Can be integrated for additional methods if needed

## Future Enhancements

Potential improvements for future versions:
- Integration with ColorTransferLib for additional methods
- GPU acceleration for batch processing
- Automatic reference image selection
- Adaptive threshold determination for divergence
- Support for other color spaces (HSV, YCbCr)
- Stain normalization specific to H&E images

## Contributing

When adding new color correction methods:
1. Add method to `ColorCorrector.correct_image()`
2. Add corresponding tests in `tests/test_color_correction.py`
3. Update this documentation
4. Ensure backward compatibility

## License

This module is part of the autoslide project and follows the same MIT license.
