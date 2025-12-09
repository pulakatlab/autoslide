# Final Annotation

The final annotation stage performs precise tissue labeling and generates high-quality masks for detailed analysis.

## Overview

This stage refines the initial annotations, integrates manual labels from annotation tools like Labelbox, and produces final segmentation masks.

## How It Works

1. **Load Initial Annotations** - Import results from initial annotation stage
2. **Import Manual Labels** - Load annotations from Labelbox or other tools
3. **Merge Annotations** - Combine automated and manual annotations
4. **Refine Segmentation** - Apply advanced segmentation algorithms
5. **Generate Masks** - Create binary masks for each tissue class
6. **Quality Control** - Validate mask quality

## Usage

```bash
python src/pipeline/annotation/final_annotation.py
```

## Integration with Labelbox

AutoSlide can import annotations from Labelbox NDJSON exports:

1. Export annotations from Labelbox in NDJSON format
2. Place exports in the annotations directory
3. Run final annotation to merge with automated results

## Output

- Multi-class tissue segmentation masks
- Labeled region images
- Annotation metadata in JSON format
- Quality metrics

## Next Steps

- [Region Suggestion](region-suggestion.md) - Extract analysis regions
- [Mask Validation GUI](../tools/mask-validation-gui.md) - Quality control
