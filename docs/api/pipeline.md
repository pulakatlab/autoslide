# Pipeline API Reference

API documentation for AutoSlide pipeline modules.

## Core Pipeline Functions

### run_pipeline.py

Main pipeline orchestration module.

```python
def run_pipeline(
    data_dir: str,
    skip_annotation: bool = False,
    config: dict = None
) -> dict:
    """
    Run the complete AutoSlide pipeline.
    
    Args:
        data_dir: Path to data directory
        skip_annotation: Skip annotation stages if True
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing pipeline results and metadata
    """
```

## Annotation Modules

### initial_annotation.py

```python
def initial_annotation(
    slide_path: str,
    output_dir: str,
    threshold_method: str = "otsu"
) -> dict:
    """
    Perform initial tissue annotation.
    
    Args:
        slide_path: Path to .svs slide file
        output_dir: Directory for output files
        threshold_method: Thresholding method ("otsu", "adaptive", "manual")
        
    Returns:
        Dictionary with annotation results
    """
```

### final_annotation.py

```python
def final_annotation(
    slide_path: str,
    initial_annotation_path: str,
    labelbox_export: str = None,
    output_dir: str = None
) -> dict:
    """
    Perform final tissue annotation with label integration.
    
    Args:
        slide_path: Path to .svs slide file
        initial_annotation_path: Path to initial annotation results
        labelbox_export: Optional path to Labelbox NDJSON export
        output_dir: Directory for output files
        
    Returns:
        Dictionary with final annotation results
    """
```

## Region Suggestion

### suggest_regions.py

```python
def suggest_regions(
    annotation_path: str,
    slide_path: str,
    output_dir: str,
    region_size: tuple = (512, 512),
    min_tissue_ratio: float = 0.5
) -> list:
    """
    Extract optimal regions for analysis.
    
    Args:
        annotation_path: Path to annotation results
        slide_path: Path to .svs slide file
        output_dir: Directory for output files
        region_size: Dimensions of extracted regions
        min_tissue_ratio: Minimum tissue content ratio
        
    Returns:
        List of extracted region metadata
    """
```

## Model Inference

### prediction.py

```python
def predict_vessels(
    image_dir: str,
    model_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    batch_size: int = 4
) -> dict:
    """
    Detect vessels using Mask R-CNN.
    
    Args:
        image_dir: Directory containing region images
        model_path: Path to model weights (.pth)
        output_dir: Directory for output files
        confidence_threshold: Minimum detection confidence
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with prediction results
    """
```

## Utility Functions

### utils.py

```python
def load_slide(slide_path: str, level: int = 0) -> np.ndarray:
    """Load slide image at specified pyramid level."""

def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save binary mask as PNG."""

def generate_section_hash(
    slide_id: str,
    coordinates: tuple,
    params: dict
) -> str:
    """Generate unique SHA-256 hash for section tracking."""

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create overlay visualization of mask on image."""
```

## Next Steps

- [Models API](models.md)
- [Utils API](utils.md)
- [Pipeline Overview](../pipeline/overview.md)
