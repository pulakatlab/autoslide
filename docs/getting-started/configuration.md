# Configuration

AutoSlide uses a simple JSON configuration file to manage pipeline settings.

## Configuration File

The main configuration file is located at `src/config.json`:

```json
{
    "data_dir": "/path/to/your/data"
}
```

## Data Directory Structure

Your data directory should be organized as follows:

```
data_dir/
├── slides/              # Raw .svs slide files
├── annotations/         # Annotation files (NDJSON, CSV)
├── models/             # Pre-trained model files (.pth)
└── output/             # Pipeline output directory
    ├── initial_annotation/
    ├── final_annotation/
    ├── regions/
    ├── predictions/
    └── fibrosis/
```

## Pipeline Parameters

### Annotation Parameters

Adjust thresholding and morphological operations in the annotation scripts:

- **Threshold values** - Control tissue detection sensitivity
- **Morphological kernel sizes** - Affect region smoothing and cleanup
- **Minimum region size** - Filter out small artifacts

### Region Suggestion Parameters

Configure region extraction in `suggest_regions.py`:

- **Region size** - Dimensions of extracted sections
- **Overlap** - Amount of overlap between adjacent regions
- **Selection criteria** - Tissue density, vessel presence, etc.

### Vessel Detection Parameters

Model inference settings in `model/prediction.py`:

- **Confidence threshold** - Minimum detection confidence
- **NMS threshold** - Non-maximum suppression threshold
- **Batch size** - Number of images processed simultaneously

### Fibrosis Quantification Parameters

HSV color analysis settings:

```bash
python src/fibrosis_calculation/calc_fibrosis.py \
    --hue-value 0.6785 \
    --hue-width 0.4 \
    --verbose
```

- **hue-value** - Target hue for fibrotic tissue
- **hue-width** - Tolerance around target hue
- **verbose** - Enable detailed logging

## Environment Variables

Set environment variables for additional configuration:

```bash
export AUTOSLIDE_DATA_DIR=/path/to/data
export AUTOSLIDE_MODEL_PATH=/path/to/model.pth
export CUDA_VISIBLE_DEVICES=0  # GPU selection
```

## Advanced Configuration

For advanced users, modify pipeline behavior directly in the source code:

- `src/pipeline/utils.py` - Core utility functions
- `src/pipeline/annotation/` - Annotation algorithms
- `src/pipeline/model/` - Model training and inference

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Pipeline Overview](../pipeline/overview.md)
- [API Reference](../api/pipeline.md)
