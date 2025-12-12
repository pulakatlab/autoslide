# Model Selection: Mask R-CNN vs UNet

This project now supports two segmentation models for vessel detection:

## Available Models

### 1. Mask R-CNN (Default)
- Instance segmentation model with ResNet-50 backbone
- Better for detecting individual vessel instances
- Provides bounding boxes and confidence scores
- More computationally intensive

### 2. UNet
- Semantic segmentation model
- Faster inference time
- Simpler architecture
- Better for dense segmentation tasks
- Based on: https://github.com/milesial/Pytorch-UNet

## Usage

### Training

Train with Mask R-CNN (default):
```bash
python -m autoslide.src.pipeline.model.training
```

Train with UNet:
```bash
python -m autoslide.src.pipeline.model.training --model unet
```

Force retraining:
```bash
python -m autoslide.src.pipeline.model.training --model unet --retrain
```

### Prediction

Predict with Mask R-CNN (default):
```bash
python -m autoslide.src.pipeline.model.prediction
```

Predict with UNet:
```bash
python -m autoslide.src.pipeline.model.prediction --model-type unet
```

With custom model path:
```bash
python -m autoslide.src.pipeline.model.prediction --model-type unet --model-path path/to/model.pth
```

## Model Files

Trained models are saved in the artifacts directory:
- Mask R-CNN: `best_val_mask_rcnn_model.pth`
- UNet: `best_val_unet_model.pth`

## Implementation Details

### UNet Architecture
- Input: 3-channel RGB images
- Output: 1-channel binary segmentation mask
- Uses bilinear upsampling
- Binary Cross-Entropy with Logits loss
- Adam optimizer with learning rate 0.001

### Mask R-CNN Architecture
- Pretrained ResNet-50 backbone with FPN
- 2 classes: background and vessel
- SGD optimizer with learning rate 0.005
- Includes region proposal network (RPN)

## Choosing a Model

Use **Mask R-CNN** when:
- You need instance-level segmentation
- You want confidence scores for predictions
- Accuracy is more important than speed

Use **UNet** when:
- You need faster inference
- Dense segmentation is sufficient
- You have limited computational resources
- You want simpler model architecture
