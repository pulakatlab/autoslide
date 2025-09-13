# 🔬 AutoSlide: AI-Powered Histological Analysis

**Unlock the hidden patterns in your histological slides with deep learning**

AutoSlide is a comprehensive pipeline that transforms how researchers analyze histological slides, combining computer vision with deep learning to identify tissues, detect vessels, and quantify fibrosis with unprecedented precision.

## ✨ Key Features

- **Automated Tissue Recognition** - Instantly identify and classify different tissue types
- **Smart Region Selection** - Automatically extract the most informative regions for analysis
- **Advanced Vessel Detection** - Precisely locate and measure blood vessels using Mask R-CNN
- **Fibrosis Quantification** - Objectively measure fibrotic changes in tissue samples
- **Reproducible Workflow** - Ensure consistent results across multiple samples and studies
- **Comprehensive Data Management** - Track annotations and regions with unique hashing system

## 🚀 The AutoSlide Pipeline

Our end-to-end workflow transforms raw histological slides into actionable insights:

1. **🔍 Initial Annotation** - Intelligent thresholding and region identification
2. **🏷️ Final Annotation** - Precise tissue labeling and mask generation  
3. **📊 Region Suggestion** - Strategic selection of analysis-ready sections
4. **🤖 Model Training** - Fine-tuned Mask R-CNN with data augmentation
5. **🔬 Vessel Detection** - Deep learning-based identification of vascular structures
6. **📈 Fibrosis Quantification** - Automated measurement of fibrotic tissue using HSV color analysis

### Initial Annotation
![TRI_85B-113_86A-118_38696](https://github.com/user-attachments/assets/5e149cdc-6469-4fe7-9c11-4e710237eb35)

### Final Annotation
<img src="https://github.com/user-attachments/assets/5976b0c1-0631-4360-8c65-9313ea431ffd" alt="Alt Text" height="800">

### Region Suggestion
<img src="https://github.com/user-attachments/assets/37600c55-e6da-4e7c-af2d-248f5ccdbb80" alt="Alt Text" height="800">

#### Examples of extracted sections
<img src="https://github.com/user-attachments/assets/315ffd0d-a0d8-4de3-b472-ae7cc939b65f" alt="Alt Text" width="300" height="300">
<img src="https://github.com/user-attachments/assets/4399c97c-00d5-4efa-9612-a6806c8d1ac0" alt="Alt Text" width="300" height="300">




## 💡 Why AutoSlide?

- **Save Time** - Automate tedious manual annotation and region selection
- **Increase Accuracy** - Leverage deep learning for consistent, objective analysis
- **Enhance Reproducibility** - Standardize your histological analysis workflow
- **Discover More** - Identify patterns and features invisible to the human eye
- **Scale Analysis** - Process multiple slides efficiently with batch processing

## 🛠️ Technical Highlights

- **Deep Learning Integration** - Fine-tuned Mask R-CNN models for precise vessel detection
- **Advanced Image Processing** - Sophisticated morphological operations for tissue segmentation
- **Intelligent Selection Algorithms** - Context-aware region extraction based on tissue properties
- **Comprehensive Visualization** - Rich visual outputs at every stage of the pipeline
- **Data Augmentation** - Negative sampling and artificial vessel generation for robust training
- **Unique Section Tracking** - SHA-256 based hashing for reproducible section identification

## 📊 Data Preparation & Model Training

AutoSlide includes specialized tools for:
- Converting annotations from Labelbox and other labeling tools into training-ready formats
- Creating high-quality binary masks from polygon annotations
- Fine-tuning state-of-the-art deep learning models on your specific tissue types
- Generating insightful visualizations of model performance
- Automated data augmentation with negative samples and artificial vessels
- Comprehensive train/validation splitting with visualization

## 📋 Requirements

The pipeline leverages powerful Python libraries:
- **slideio** - For efficient slide image handling (.svs format support)
- **PyTorch** - For deep learning model training and inference
- **torchvision** - For computer vision models and transforms
- **scikit-learn** - For clustering and dimensionality reduction
- **OpenCV** - For advanced image processing
- **matplotlib/pandas** - For visualization and data handling
- **PIL/Pillow** - For image manipulation
- **tqdm** - For progress tracking
- **numpy/scipy** - For numerical computations

## 🔧 Getting Started

### Quick Start
Run the complete pipeline with a single command:
```bash
python autoslide/pipeline/run_pipeline.py
```

### Configuration
Set up your data directory in `autoslide/config.json`:
```json
{
    "data_dir": "/path/to/your/data"
}
```

### Individual Pipeline Steps
Execute specific stages as needed:

```bash
# Initial annotation only
python autoslide/pipeline/annotation/initial_annotation.py

# Final annotation only  
python autoslide/pipeline/annotation/final_annotation.py

# Region suggestion
python autoslide/pipeline/suggest_regions.py

# Model training with options
python autoslide/pipeline/model/training.py --retrain

# Prediction
python autoslide/pipeline/model/prediction.py

# Fibrosis quantification
python autoslide/fibrosis_calculation/calc_fibrosis.py
```

### Command Line Options
```bash
# Skip annotation steps
python autoslide/pipeline/run_pipeline.py --skip_annotation

# Skip training (use existing model)
python autoslide/pipeline/run_pipeline.py --skip_training

# Disable data augmentation
python autoslide/pipeline/run_pipeline.py --no_augmentation

# Fibrosis quantification with custom parameters
python autoslide/fibrosis_calculation/calc_fibrosis.py --hue-value 0.6785 --hue-width 0.4 --verbose
```

## 🗂️ Project Structure

```
autoslide/
├── pipeline/
│   ├── annotation/          # Tissue annotation modules
│   ├── model/              # Deep learning training and prediction
│   ├── label_handling/     # Label import/export utilities
│   └── utils.py           # Core utility functions
├── utils/                  # Additional utilities
├── fibrosis_calculation/   # Fibrosis quantification tools
│   └── calc_fibrosis.py   # Main fibrosis analysis module
└── config.json            # Configuration file
```

## 📈 Output and Results

AutoSlide generates comprehensive outputs including:
- **Annotated slide visualizations** with tissue boundaries and labels
- **Region selection maps** showing extracted analysis areas
- **Model training metrics** and loss curves
- **Prediction visualizations** with detected vessels highlighted
- **Fibrosis quantification reports** with HSV-based percentage measurements and visualizations
- **Section tracking data** with unique identifiers for reproducibility

## 🔬 Supported File Formats

- **Input**: .svs slide files (Aperio format)
- **Annotations**: Labelbox NDJSON exports, CSV metadata
- **Models**: PyTorch .pth files
- **Outputs**: PNG images, CSV data, JSON tracking files

Ready to transform your histological analysis? Get started with AutoSlide today!

## 📚 Using DVC for Model and Data Versioning

<details>
<summary>Click to expand DVC setup instructions</summary>

This project uses [DVC (Data Version Control)](https://dvc.org/) to track large files and model artifacts.

### Setup

1. Install DVC:
```bash
pip install dvc
# For Google Drive storage
pip install dvc[gdrive]
# For S3 storage
pip install dvc[s3]
```

2. Initialize DVC in your repository (already done):
```bash
dvc init
```

3. Configure remote storage:
```bash
# For Google Drive
dvc remote add -d myremote gdrive://path/to/folder
# For S3
dvc remote add -d myremote s3://bucket/path
```

### Working with DVC

#### Adding files to DVC

```bash
# Add a large file or directory to DVC
dvc add data/large_dataset.svs
dvc add artifacts/mask_rcnn_model.pth
```

This creates a small .dvc file that you commit to git instead of the large file.

#### Pushing and pulling data

```bash
# Push your data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

#### Versioning models

When you train a new model version:

```bash
# Add the new model file
dvc add artifacts/mask_rcnn_model.pth

# Commit the .dvc file to git
git add artifacts/mask_rcnn_model.pth.dvc
git commit -m "Update model with improved accuracy"

# Push the actual model file to remote storage
dvc push
```

#### Switching between versions

```bash
# Checkout a specific git commit/tag
git checkout <commit-hash>

# Get the corresponding data files
dvc pull
```

### Best Practices

1. Always add large files with DVC, not git
2. Commit .dvc files to git after running `dvc add`
3. Run `dvc push` after adding new files to make them available to others
4. Use `dvc pull` after checkout to get the correct version of data files

For more information, visit the [DVC documentation](https://dvc.org/doc).

</details>
