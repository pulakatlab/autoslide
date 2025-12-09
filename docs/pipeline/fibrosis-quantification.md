# Fibrosis Quantification

The fibrosis quantification stage measures fibrotic tissue content using HSV color space analysis.

## Overview

This stage analyzes tissue sections to quantify fibrosis based on color characteristics, providing objective percentage measurements of fibrotic content.

## How It Works

1. **Load Region Images** - Import extracted tissue sections
2. **Convert to HSV** - Transform from RGB to HSV color space
3. **Define Fibrosis Range** - Set hue parameters for fibrotic tissue
4. **Create Mask** - Identify pixels matching fibrosis criteria
5. **Calculate Percentage** - Compute fibrotic area relative to total tissue
6. **Generate Visualizations** - Create overlay images showing fibrotic regions

## Usage

### Basic Usage

```bash
python src/fibrosis_calculation/calc_fibrosis.py
```

### With Custom Parameters

```bash
python src/fibrosis_calculation/calc_fibrosis.py \
    --hue-value 0.6785 \
    --hue-width 0.4 \
    --verbose
```

## Parameters

### Hue Value

The target hue for fibrotic tissue (0.0 - 1.0):

- Default: `0.6785` (blue-purple range typical of fibrosis staining)
- Adjust based on your staining protocol

### Hue Width

Tolerance around the target hue (0.0 - 1.0):

- Default: `0.4`
- Larger values capture more color variation
- Smaller values are more selective

### Saturation and Value Thresholds

Additional filters to refine detection:

- Minimum saturation to exclude pale regions
- Minimum value to exclude dark artifacts

## Output

- Fibrosis percentage for each region
- Visualization overlays highlighting fibrotic areas
- CSV file with quantification results
- Summary statistics

## Validation

Results can be validated using:

1. Manual review of overlay images
2. Comparison with pathologist assessments
3. Statistical analysis across sample cohorts

## Tips for Accurate Quantification

- **Calibrate parameters** - Adjust hue values for your specific staining
- **Quality control** - Use the mask validation GUI to review results
- **Consistent staining** - Ensure uniform staining across samples
- **Exclude artifacts** - Remove regions with staining artifacts before quantification

## Next Steps

- [Mask Validation GUI](../tools/mask-validation-gui.md) - Review quantification results
- [Color Correction](../advanced/color-correction.md) - Normalize staining variations
