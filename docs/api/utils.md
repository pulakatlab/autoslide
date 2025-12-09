# Utils API Reference

API documentation for AutoSlide utility functions.

## Image Processing

### Image Loading

```python
def load_slide(
    slide_path: str,
    level: int = 0,
    region: tuple = None
) -> np.ndarray:
    """
    Load whole slide image or region.
    
    Args:
        slide_path: Path to .svs file
        level: Pyramid level (0 = highest resolution)
        region: Optional (x, y, width, height) tuple
        
    Returns:
        Image as numpy array
    """
```

### Image Preprocessing

```python
def preprocess_image(
    image: np.ndarray,
    target_size: tuple = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image
        target_size: Optional resize dimensions
        normalize: Apply normalization if True
        
    Returns:
        Preprocessed image
    """
```

## Mask Operations

### Mask Generation

```python
def create_binary_mask(
    polygons: list,
    image_shape: tuple
) -> np.ndarray:
    """
    Create binary mask from polygon annotations.
    
    Args:
        polygons: List of polygon coordinates
        image_shape: Output mask dimensions
        
    Returns:
        Binary mask as numpy array
    """
```

### Mask Visualization

```python
def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay visualization.
    
    Args:
        image: Base image
        mask: Binary mask
        color: Overlay color (RGB)
        alpha: Transparency (0-1)
        
    Returns:
        Overlay image
    """
```

## Section Tracking

### Hash Generation

```python
def generate_section_hash(
    slide_id: str,
    coordinates: tuple,
    params: dict = None
) -> str:
    """
    Generate unique SHA-256 hash for section.
    
    Args:
        slide_id: Slide identifier
        coordinates: (x, y, width, height) tuple
        params: Optional extraction parameters
        
    Returns:
        SHA-256 hash string
    """
```

### Metadata Management

```python
def save_section_metadata(
    sections: list,
    output_path: str
) -> None:
    """
    Save section tracking metadata to JSON.
    
    Args:
        sections: List of section dictionaries
        output_path: Output JSON file path
    """

def load_section_metadata(
    metadata_path: str
) -> list:
    """
    Load section tracking metadata from JSON.
    
    Args:
        metadata_path: Path to metadata JSON
        
    Returns:
        List of section dictionaries
    """
```

## File I/O

### Save Functions

```python
def save_image(
    image: np.ndarray,
    output_path: str,
    quality: int = 95
) -> None:
    """Save image to file."""

def save_mask(
    mask: np.ndarray,
    output_path: str
) -> None:
    """Save binary mask as PNG."""

def save_results_csv(
    results: list,
    output_path: str
) -> None:
    """Save results to CSV file."""
```

## Color Space Conversions

```python
def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV color space."""

def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert HSV image to RGB color space."""
```

## Morphological Operations

```python
def apply_morphology(
    mask: np.ndarray,
    operation: str = "close",
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply morphological operation to mask.
    
    Args:
        mask: Binary mask
        operation: "open", "close", "dilate", or "erode"
        kernel_size: Size of structuring element
        
    Returns:
        Processed mask
    """
```

## Next Steps

- [Pipeline API](pipeline.md)
- [Models API](models.md)
- [Pipeline Overview](../pipeline/overview.md)
