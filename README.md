# 🔬 AutoSlide: AI-Powered Histological Analysis

**Unlock the hidden patterns in your histological slides with deep learning**

AutoSlide is a comprehensive pipeline that transforms how researchers analyze histological slides, combining computer vision with deep learning to identify tissues, detect vessels, and quantify fibrosis with unprecedented precision.

## ✨ Key Features

- **Automated Tissue Recognition** - Instantly identify and classify different tissue types
- **Smart Region Selection** - Automatically extract the most informative regions for analysis
- **Advanced Vessel Detection** - Precisely locate and measure blood vessels using Mask R-CNN
- **Fibrosis Quantification** - Objectively measure fibrotic changes in tissue samples
- **Reproducible Workflow** - Ensure consistent results across multiple samples and studies

## 🚀 The AutoSlide Pipeline

Our end-to-end workflow transforms raw histological slides into actionable insights:

1. **🔍 Initial Annotation** - Intelligent thresholding and region identification
2. **🏷️ Final Annotation** - Precise tissue labeling and mask generation
3. **📊 Region Suggestion** - Strategic selection of analysis-ready sections
4. **🧩 Pixel Clustering** - Advanced segmentation of tissue components
5. **🔬 Vessel Detection** - Deep learning-based identification of vascular structures

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

## 🛠️ Technical Highlights

- **Deep Learning Integration** - Fine-tuned Mask R-CNN models for precise vessel detection
- **Advanced Image Processing** - Sophisticated morphological operations for tissue segmentation
- **Intelligent Selection Algorithms** - Context-aware region extraction based on tissue properties
- **Comprehensive Visualization** - Rich visual outputs at every stage of the pipeline

## 📊 Data Preparation & Model Training

AutoSlide includes specialized tools for:
- Converting annotations from popular labeling tools into training-ready formats
- Creating high-quality binary masks from polygon annotations
- Fine-tuning state-of-the-art deep learning models on your specific tissue types
- Generating insightful visualizations of model performance

## 📋 Requirements

The pipeline leverages powerful Python libraries:
- **slideio** - For efficient slide image handling
- **PyTorch** - For deep learning model training and inference
- **scikit-learn** - For clustering and dimensionality reduction
- **OpenCV** - For advanced image processing
- **matplotlib/pandas** - For visualization and data handling

## 🔧 Getting Started

Run the complete pipeline with a single command:
```bash
python src/pipeline/run_pipeline.py
```

Or execute individual stages as needed for your specific workflow.

Ready to transform your histological analysis? Get started with AutoSlide today!

<details>
<summary>## Using DVC for Model and Data Versioning</summary>

# AutoSlide Artifacts

This directory contains model artifacts and other large files used by the AutoSlide project.

## Using DVC for Model and Data Versioning

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
