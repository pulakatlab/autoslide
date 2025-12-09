# Histogram Percentile Color Normalization

## Overview

This module implements histogram percentile-based color normalization for histological images. This method is particularly effective for normalizing staining intensity variations across different batches or scanning sessions.

## Features

- **Percentile-based normalization**: Maps histogram percentiles between reference and target images
- **Robust to outliers**: Uses percentile clipping to handle extreme values
- **Batch processing**: Efficiently process entire directories
- **Histogram comparison**: Utilities to compare color distributions
- **Customizable percentiles**: Adjust percentile range for different use cases

## Installation

No additional dependencies required! Uses existing dependencies:
- `numpy >= 1.20.0`
- `opencv-python >= 4.5.0`

## Usage

### Basic Usage

```python
from autoslide.pipeline.model.color_normalization import HistogramPercentileNormalizer
import cv2

# Initialize with reference image(s)
normalizer = HistogramPercentileNormalizer('path/to/reference_image.png')

# Load and normalize an image
img = cv2.imread('new_dataset/image.png')
normalized = normalizer.normalize_image(img)
cv2.imwrite('normalized_image.png', normalized)
```

### Custom Percentile Range

```python
# Use different percentile range (default is 1.0, 99.0)
normalizer = HistogramPercentileNormalizer(
    'reference.png',
    percentiles=(5.0, 95.0)  # More aggressive clipping
)

normalized = normalizer.normalize_image(img)
```

### Multiple Reference Images

```python
# Use multiple reference images for more robust statistics
normalizer = HistogramPercentileNormalizer([
    'original_dataset/sample1.png',
    'original_dataset/sample2.png',
    'original_dataset/sample3.png'
])

normalized = normalizer.normalize_image(img)
```

### Batch Processing

```python
from autoslide.pipeline.model.color_normalization import batch_histogram_normalization

# Normalize all images in a directory
successful, failed = batch_histogram_normalization(
    input_dir='data/new_dataset/images',
    output_dir='data/new_dataset/normalized',
    reference_images=['data/original/ref1.png', 'data/original/ref2.png'],
    percentiles=(1.0, 99.0),
    file_pattern='*.png'
)

print(f"Normalized {successful} images, {failed} failed")
```

### Histogram Comparison

```python
from autoslide.pipeline.model.color_normalization import compare_histograms
import cv2

# Compare histograms between two images
img1 = cv2.imread('dataset1/image.png')
img2 = cv2.imread('dataset2/image.png')

metrics = compare_histograms(img1, img2)

for channel in ['channel_0', 'channel_1', 'channel_2']:
    print(f"{channel}:")
    print(f"  Correlation: {metrics[channel]['correlation']:.4f}")
    print(f"  Chi-square: {metrics[channel]['chi_square']:.4f}")
    print(f"  Intersection: {metrics[channel]['intersection']:.4f}")
```

## How It Works

### Algorithm

The histogram percentile normalization method works as follows:

1. **Compute reference percentiles**: For each color channel in the reference image(s), compute the values at the specified percentiles (e.g., 1st and 99th percentiles).

2. **Compute target percentiles**: For each color channel in the target image, compute the same percentiles.

3. **Linear mapping**: Map the target percentile range to the reference percentile range using linear transformation:
   ```
   out = (in - low_target) / (high_target - low_target) * (high_ref - low_ref) + low_ref
   ```

4. **Clipping**: Clip values to the valid range [0, 255].

### Why Percentiles?

- **Robust to outliers**: Unlike using min/max values, percentiles are not affected by extreme outliers
- **Preserves structure**: The relative relationships between pixel intensities are maintained
- **Handles staining variations**: Effectively normalizes differences in staining intensity
- **Customizable**: Can adjust percentile range based on data characteristics

### Comparison with Other Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Histogram Percentile** | Robust to outliers, simple, fast | May not preserve exact color relationships | Staining intensity variations |
| Reinhard Color Transfer | Preserves color relationships, works in LAB space | More complex, slower | General color correction |
| Histogram Matching | Exact histogram matching | Sensitive to outliers | Similar image content |

## Use Cases

### 1. Normalizing Staining Intensity

When different tissue batches are stained with varying intensity:

```python
normalizer = HistogramPercentileNormalizer(
    'well_stained_reference.png',
    percentiles=(1.0, 99.0)
)

# Normalize under-stained images
normalized = normalizer.normalize_image(under_stained_img)
```

### 2. Scanner Calibration

When images from different scanners have different intensity ranges:

```python
# Use images from Scanner A as reference
normalizer = HistogramPercentileNormalizer([
    'scanner_a/image1.png',
    'scanner_a/image2.png'
])

# Normalize images from Scanner B
normalized = normalizer.normalize_image(scanner_b_img)
```

### 3. Quality Control

Check if normalization is needed:

```python
from autoslide.pipeline.model.color_normalization import compare_histograms

ref_img = cv2.imread('reference.png')
new_img = cv2.imread('new_image.png')

metrics = compare_histograms(ref_img, new_img)

# Check correlation for each channel
for channel in ['channel_0', 'channel_1', 'channel_2']:
    correlation = metrics[channel]['correlation']
    if correlation < 0.8:  # Threshold can be adjusted
        print(f"⚠️  {channel}: Low correlation ({correlation:.3f}) - normalization recommended")
    else:
        print(f"✅ {channel}: Good correlation ({correlation:.3f})")
```

## Integration with Training Pipeline

### Before Training

```python
from autoslide.pipeline.model.color_normalization import batch_histogram_normalization

# Normalize new dataset before training
print("Applying histogram percentile normalization...")
batch_histogram_normalization(
    input_dir='data/new_dataset/images',
    output_dir='data/new_dataset/normalized',
    reference_images=[
        'data/original_dataset/sample1.png',
        'data/original_dataset/sample2.png'
    ],
    percentiles=(1.0, 99.0)
)

# Then proceed with training using normalized images
# ... rest of training code ...
```

### During Prediction

```python
from autoslide.pipeline.model.color_normalization import HistogramPercentileNormalizer

# Initialize normalizer with training dataset references
normalizer = HistogramPercentileNormalizer(
    ['training_data/ref1.png', 'training_data/ref2.png']
)

# Normalize images before prediction
img = cv2.imread('new_image.png')
normalized = normalizer.normalize_image(img)

# Perform prediction on normalized image
# ... prediction code ...
```

## API Reference

### HistogramPercentileNormalizer Class

```python
class HistogramPercentileNormalizer:
    def __init__(
        self,
        reference_images: Union[str, Path, List[Union[str, Path]]],
        percentiles: Tuple[float, float] = (1.0, 99.0)
    )
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray
    
    def normalize_image_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool
    
    def get_image_percentiles(self, image: np.ndarray) -> dict
```

### Functions

```python
def batch_histogram_normalization(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reference_images: Union[str, Path, List[Union[str, Path]]],
    percentiles: Tuple[float, float] = (1.0, 99.0),
    file_pattern: str = '*.png'
) -> Tuple[int, int]

def compare_histograms(
    image1: np.ndarray,
    image2: np.ndarray,
    bins: int = 256
) -> dict
```

## Testing

The module includes comprehensive tests in `tests/test_color_normalization.py`:

```bash
# Run tests (requires pytest and dependencies installed)
pytest tests/test_color_normalization.py -v
```

Test coverage includes:
- Normalizer initialization
- Image normalization
- Batch processing
- Histogram comparison
- Integration scenarios
- Edge cases (uniform images, outliers)

## Performance

- **Processing time**: ~5-20ms per 512×512 image (CPU)
- **Memory usage**: ~2× image size (for percentile computation)
- **Scalability**: Linear with number of images

## Choosing Percentile Range

| Percentile Range | Use Case | Effect |
|------------------|----------|--------|
| (0.5, 99.5) | Very robust | Clips more outliers, may lose some detail |
| **(1.0, 99.0)** | **Default, recommended** | **Good balance** |
| (2.0, 98.0) | Aggressive clipping | Handles extreme outliers well |
| (5.0, 95.0) | Very aggressive | May over-normalize |

## Troubleshooting

### Issue: Normalized images look washed out

Try using a narrower percentile range:
```python
normalizer = HistogramPercentileNormalizer(
    'reference.png',
    percentiles=(2.0, 98.0)  # More aggressive clipping
)
```

### Issue: Colors still look different

- Ensure reference images are representative of desired appearance
- Try using multiple reference images for more robust statistics
- Consider using Reinhard color transfer instead (in `color_correction.py`)

### Issue: Normalization too aggressive

Use a wider percentile range:
```python
normalizer = HistogramPercentileNormalizer(
    'reference.png',
    percentiles=(0.5, 99.5)  # Less aggressive
)
```

## Comparison with Reinhard Method

Both methods are available in the autoslide package:

**Histogram Percentile** (this module):
- Simpler, faster
- Works in RGB/BGR space
- Best for intensity normalization
- More robust to outliers

**Reinhard Color Transfer** (`color_correction.py`):
- More sophisticated
- Works in LAB color space
- Best for color characteristic transfer
- Preserves perceptual color relationships

Choose based on your specific needs:
- Use **Histogram Percentile** for staining intensity variations
- Use **Reinhard** for general color correction and color space transformations

## References

1. Histogram equalization and percentile-based methods are standard techniques in image processing
2. Commonly used in medical image analysis for intensity normalization
3. Related to histogram matching and specification techniques

## Future Enhancements

Potential improvements:
- Adaptive percentile selection based on image content
- Per-tissue-type normalization
- Integration with stain separation methods
- GPU acceleration for batch processing

## Contributing

When adding new normalization methods:
1. Add method to `HistogramPercentileNormalizer` class or create new class
2. Add corresponding tests in `tests/test_color_normalization.py`
3. Update this documentation
4. Ensure backward compatibility

## License

This module is part of the autoslide project and follows the same MIT license.
