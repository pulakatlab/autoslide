# Initial Annotation

The initial annotation stage performs automated tissue detection and region identification in whole slide images.

## Overview

This stage uses adaptive thresholding and morphological operations to identify tissue regions in .svs slide files, providing the foundation for downstream analysis.

## How It Works

1. **Slide Loading** - Load whole slide image at appropriate resolution
2. **Preprocessing** - Convert to grayscale and apply noise reduction
3. **Thresholding** - Adaptive thresholding to separate tissue from background
4. **Morphological Operations** - Clean up detected regions using opening/closing
5. **Contour Detection** - Extract region boundaries
6. **Visualization** - Generate annotated images showing detected regions

## Usage

```bash
python src/pipeline/annotation/initial_annotation.py
```

## Parameters

Key parameters that can be adjusted in the source code:

- **Threshold method** - Otsu, adaptive, or manual threshold
- **Kernel size** - Size of morphological operation kernels
- **Minimum area** - Minimum region size to keep
- **Resolution level** - Slide pyramid level for processing

## Output

- Annotated slide images with detected regions highlighted
- Binary masks of tissue regions
- Region metadata (coordinates, areas, etc.)

## Next Steps

- [Final Annotation](final-annotation.md) - Refine tissue segmentation
- [Region Suggestion](region-suggestion.md) - Extract analysis regions
