# Vessel Detection

The vessel detection stage uses deep learning to identify and segment blood vessels in tissue sections.

## Overview

This stage applies a pre-trained Mask R-CNN model to detect vascular structures with instance segmentation, providing precise vessel boundaries and locations.

## How It Works

1. **Load Pre-trained Model** - Import Mask R-CNN weights
2. **Prepare Images** - Preprocess extracted regions for inference
3. **Run Inference** - Detect vessels using the model
4. **Post-processing** - Apply confidence thresholding and NMS
5. **Generate Visualizations** - Create overlay images showing detections
6. **Save Results** - Export predictions in JSON format

## Usage

```bash
python src/pipeline/model/prediction.py
```

## Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Detection Head**: Mask R-CNN for instance segmentation
- **Training Data**: Histological slides with manually annotated vessels

## Parameters

Adjust detection parameters:

```python
# Confidence threshold
confidence_threshold = 0.5

# Non-maximum suppression threshold
nms_threshold = 0.3

# Batch size for inference
batch_size = 4
```

## Output

- Predicted vessel masks for each region
- Bounding boxes and confidence scores
- Visualization overlays
- JSON file with detection metadata

## Performance

The pre-trained model achieves:

- High precision on vessel detection
- Robust performance across different tissue types
- Fast inference suitable for batch processing

## Advanced Usage

For custom predictions on arbitrary directories, see [Arbitrary Directory Prediction](../advanced/arbitrary-directory-prediction.md).

## Next Steps

- [Fibrosis Quantification](fibrosis-quantification.md) - Measure fibrosis
- [Mask Validation GUI](../tools/mask-validation-gui.md) - Review predictions
