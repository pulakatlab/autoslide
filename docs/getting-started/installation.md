# Installation

## Requirements

AutoSlide requires Python 3.8 or higher and leverages powerful Python libraries for image processing and deep learning.

### Core Dependencies

- **slideio** - For efficient slide image handling (.svs format support)
- **PyTorch** - For deep learning model training and inference
- **torchvision** - For computer vision models and transforms
- **scikit-learn** - For clustering and dimensionality reduction
- **OpenCV** - For advanced image processing
- **matplotlib/pandas** - For visualization and data handling
- **PIL/Pillow** - For image manipulation
- **tqdm** - For progress tracking
- **numpy/scipy** - For numerical computations

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/pulakatlab/autoslide.git
cd autoslide
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python -c "import autoslide; print('AutoSlide installed successfully!')"
```

## Optional: DVC Setup

If you need to work with versioned data and models, install DVC:

```bash
pip install dvc

# For Google Drive storage
pip install dvc[gdrive]

# For S3 storage
pip install dvc[s3]
```

See the [DVC Setup Guide](../data/dvc-setup.md) for more details.

## Troubleshooting

### SlideIO Installation Issues

If you encounter issues installing slideio, ensure you have the necessary system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libopenslide-dev
```

**macOS:**
```bash
brew install openslide
```

### PyTorch Installation

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Visit [PyTorch's website](https://pytorch.org/get-started/locally/) for platform-specific instructions.

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration](configuration.md)
- [Pipeline Overview](../pipeline/overview.md)
