# Arbitrary Directory Prediction

## Overview

This feature allows the neural network to perform predictions on any directory containing images, not just the configured `suggested_regions` directory. This addresses issue #82 and provides flexibility for processing images from various sources.

## Features

- **Flexible input**: Process any directory containing images
- **Custom output location**: Specify where predictions should be saved
- **Multiple image formats**: Supports PNG, JPG, JPEG, TIF, TIFF
- **Visualization support**: Optional overlay generation for quality control
- **Progress tracking**: Real-time progress updates during processing

## Usage

### Basic Usage

Process all images in a directory:

```bash
python -m autoslide.pipeline.model.prediction --dir /path/to/images
```

This will:
- Load images from `/path/to/images`
- Save predictions to `/path/to/images/predictions/masks/`
- Display progress and summary

### Custom Output Directory

Specify a custom output location:

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /path/to/images \
    --output-dir /path/to/results
```

Predictions will be saved to:
- Masks: `/path/to/results/masks/`
- Overlays (if enabled): `/path/to/results/overlays/`

### With Visualizations

Generate overlay visualizations for quality control:

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /path/to/images \
    --save-visualizations
```

This creates overlay images showing predictions on top of original images.

### Custom Model

Use a specific model file:

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /path/to/images \
    --model-path /path/to/custom_model.pth
```

### Limit Processing

Process only a subset of images (useful for testing):

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /path/to/images \
    --max-images 10
```

### Verbose Output

Get detailed information during processing:

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /path/to/images \
    --verbose
```

## Complete Example

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /data/external_dataset/images \
    --output-dir /data/external_dataset/predictions \
    --model-path artifacts/best_model.pth \
    --save-visualizations \
    --max-images 100 \
    --verbose
```

## Output Structure

When using `--dir`, the output structure is:

```
<output-dir>/
├── masks/
│   ├── image1_mask.png
│   ├── image2_mask.png
│   └── ...
└── overlays/  (if --save-visualizations is used)
    ├── image1_overlay.png
    ├── image2_overlay.png
    └── ...
```

## Supported Image Formats

The following image formats are automatically detected:
- PNG (`.png`, `.PNG`)
- JPEG (`.jpg`, `.JPG`, `.jpeg`, `.JPEG`)
- TIFF (`.tif`, `.TIF`, `.tiff`, `.TIFF`)

## Use Cases

### 1. External Dataset Processing

Process images from a collaborator or external source:

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /data/collaborator_images \
    --output-dir /data/collaborator_results
```

### 2. Quality Control

Generate visualizations for a sample of images:

```bash
python -m autoslide.pipeline.model.prediction \
    --dir /data/validation_set \
    --save-visualizations \
    --max-images 50
```

### 3. Batch Processing

Process multiple directories in a script:

```bash
#!/bin/bash
for dir in /data/batch_*; do
    python -m autoslide.pipeline.model.prediction \
        --dir "$dir/images" \
        --output-dir "$dir/predictions"
done
```

### 4. Model Comparison

Compare different models on the same dataset:

```bash
# Model 1
python -m autoslide.pipeline.model.prediction \
    --dir /data/test_set \
    --output-dir /results/model1 \
    --model-path models/model1.pth

# Model 2
python -m autoslide.pipeline.model.prediction \
    --dir /data/test_set \
    --output-dir /results/model2 \
    --model-path models/model2.pth
```

## Integration with Color Normalization

Combine with histogram percentile normalization for best results:

```bash
# Step 1: Normalize colors
python -c "
from autoslide.pipeline.model.color_normalization import batch_histogram_normalization
batch_histogram_normalization(
    input_dir='/data/new_dataset/images',
    output_dir='/data/new_dataset/normalized',
    reference_images=['training_data/ref1.png', 'training_data/ref2.png']
)
"

# Step 2: Run predictions on normalized images
python -m autoslide.pipeline.model.prediction \
    --dir /data/new_dataset/normalized \
    --output-dir /data/new_dataset/predictions
```

## Python API

You can also use the functionality programmatically:

```python
from autoslide.pipeline.model.prediction import process_arbitrary_directory

process_arbitrary_directory(
    input_dir='/path/to/images',
    output_dir='/path/to/results',
    model_path='artifacts/best_model.pth',
    save_visualizations=True,
    max_images=None,  # Process all images
    verbose=True
)
```

## Comparison with Default Mode

| Feature | Default Mode | Arbitrary Directory Mode |
|---------|--------------|--------------------------|
| Input source | Config `suggested_regions` | Any directory via `--dir` |
| Output location | Alongside input images | Customizable via `--output-dir` |
| Directory structure | Preserves SVS structure | Flat structure |
| Reprocessing | `--reprocess` flag | Always processes all images |
| Use case | Standard pipeline | External datasets, testing |

## Command Line Arguments

### New Arguments (Issue #82)

- `--dir <path>`: Directory containing images to process
- `--output-dir <path>`: Output directory for predictions (default: `<dir>/predictions`)

### Existing Arguments (Still Available)

- `--model-path <path>`: Path to saved model
- `--save-visualizations`: Save prediction visualizations
- `--max-images <n>`: Maximum number of images to process
- `--verbose`, `-v`: Print detailed information
- `--reprocess`: (Only for default mode) Remove existing outputs and reprocess

## Error Handling

The script handles various error conditions gracefully:

- **No images found**: Displays message and exits
- **Model loading failure**: Shows error and exits
- **Individual image failures**: Logs error, continues with remaining images
- **Invalid paths**: Validates paths and shows helpful error messages

## Performance Considerations

- **Memory usage**: Processes images one at a time to minimize memory footprint
- **GPU acceleration**: Automatically uses GPU if available
- **Batch size**: Single image processing for arbitrary directories (no batching)
- **Progress tracking**: Uses tqdm for real-time progress updates

## Tips and Best Practices

1. **Test first**: Use `--max-images 10` to test on a small subset
2. **Check visualizations**: Use `--save-visualizations` for quality control
3. **Normalize colors**: Apply color normalization before prediction for best results
4. **Organize outputs**: Use descriptive output directory names
5. **Monitor progress**: Use `--verbose` for detailed logging

## Troubleshooting

### Issue: No images found

**Cause**: Directory doesn't contain supported image formats

**Solution**: Check that images have supported extensions (.png, .jpg, .tif, etc.)

### Issue: Out of memory

**Cause**: Images are too large or GPU memory is limited

**Solution**: 
- Process fewer images at a time using `--max-images`
- Use CPU instead of GPU (set `CUDA_VISIBLE_DEVICES=""`)

### Issue: Predictions look incorrect

**Cause**: Color distribution differs from training data

**Solution**: Apply color normalization before prediction:
```bash
# Normalize first
python -c "from autoslide.pipeline.model.color_normalization import batch_histogram_normalization; batch_histogram_normalization('input', 'normalized', 'ref.png')"

# Then predict
python -m autoslide.pipeline.model.prediction --dir normalized
```

### Issue: Slow processing

**Cause**: CPU processing or large images

**Solution**:
- Ensure GPU is available and being used
- Check image sizes (very large images take longer)
- Use `--verbose` to see per-image timing

## Examples

### Example 1: Quick Test

```bash
# Test on 5 images with visualizations
python -m autoslide.pipeline.model.prediction \
    --dir test_images \
    --max-images 5 \
    --save-visualizations \
    --verbose
```

### Example 2: Production Processing

```bash
# Process entire dataset with custom model
python -m autoslide.pipeline.model.prediction \
    --dir /data/production/images \
    --output-dir /data/production/predictions \
    --model-path /models/production_v2.pth
```

### Example 3: Batch Script

```bash
#!/bin/bash
# Process multiple datasets

DATASETS=(
    "/data/dataset1"
    "/data/dataset2"
    "/data/dataset3"
)

MODEL="/models/best_model.pth"

for dataset in "${DATASETS[@]}"; do
    echo "Processing $dataset..."
    python -m autoslide.pipeline.model.prediction \
        --dir "$dataset/images" \
        --output-dir "$dataset/predictions" \
        --model-path "$MODEL" \
        --save-visualizations
done

echo "All datasets processed!"
```

## Future Enhancements

Potential improvements:
- Recursive directory processing
- Parallel processing for multiple images
- Support for additional image formats
- Automatic color normalization option
- Batch prediction for efficiency
- Progress saving and resumption

## Related Documentation

- [Histogram Percentile Normalization](HISTOGRAM_PERCENTILE_NORMALIZATION.md)
- [Color Correction](COLOR_CORRECTION.md)
- [Model Training](../autoslide/pipeline/model/training.py)

## Contributing

When extending this feature:
1. Maintain backward compatibility with default mode
2. Add tests for new functionality
3. Update this documentation
4. Follow existing code patterns

## License

This feature is part of the autoslide project and follows the same MIT license.
