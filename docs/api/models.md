# Models API Reference

API documentation for AutoSlide model training and inference modules.

## Model Architecture

### Mask R-CNN

The vessel detection model uses Mask R-CNN with ResNet-50 backbone.

```python
def get_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Get Mask R-CNN model for vessel detection.
    
    Args:
        num_classes: Number of classes (background + vessel)
        
    Returns:
        Mask R-CNN model instance
    """
```

## Training

### train.py

```python
def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int = 50,
    learning_rate: float = 0.005,
    batch_size: int = 4,
    output_dir: str = "artifacts/"
) -> dict:
    """
    Train vessel detection model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Training batch size
        output_dir: Directory to save model checkpoints
        
    Returns:
        Dictionary with training metrics
    """
```

## Inference

### prediction.py

```python
def load_model(model_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load pre-trained model for inference.
    
    Args:
        model_path: Path to model weights (.pth)
        device: Device for inference ("cuda" or "cpu")
        
    Returns:
        Loaded model in eval mode
    """

def predict_batch(
    model: torch.nn.Module,
    images: list,
    confidence_threshold: float = 0.5
) -> list:
    """
    Run inference on batch of images.
    
    Args:
        model: Loaded model instance
        images: List of image tensors
        confidence_threshold: Minimum detection confidence
        
    Returns:
        List of prediction dictionaries
    """
```

## Data Augmentation

### augmentation.py

```python
def augment_training_data(
    images: list,
    masks: list,
    augmentation_factor: int = 3
) -> tuple:
    """
    Apply data augmentation to training samples.
    
    Args:
        images: List of training images
        masks: List of corresponding masks
        augmentation_factor: Number of augmented versions per sample
        
    Returns:
        Tuple of (augmented_images, augmented_masks)
    """
```

## Evaluation

### evaluate.py

```python
def evaluate_model(
    model: torch.nn.Module,
    test_dataset: Dataset,
    iou_threshold: float = 0.5
) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Model to evaluate
        test_dataset: Test dataset
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Dictionary with evaluation metrics (mAP, precision, recall)
    """
```

## Next Steps

- [Pipeline API](pipeline.md)
- [Utils API](utils.md)
- [Vessel Detection](../pipeline/vessel-detection.md)
