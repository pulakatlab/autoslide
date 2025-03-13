# Automated Slide Analysis Pipeline

This project provides a pipeline for automated analysis of histological slides, particularly focusing on tissue identification, region extraction, and fibrosis analysis.

## Overview

The pipeline processes histological slides (SVS format) through several stages:
1. Initial thresholding to extract tissue samples
2. Region identification and annotation
3. Extraction of sections with specific properties
4. Analysis of selected regions
5. Vessel detection using Mask R-CNN

## Pipeline Stages

### 1. Initial Annotation (`initial_annotation.py`)
- Loads slide images and performs thresholding
- Identifies distinct regions using morphological operations
- Extracts region properties (area, eccentricity, dimensions)
- Outputs regions for manual labeling
- Generates visualization of identified regions

### 2. Final Annotation (`final_annotation.py`)
- Takes manually annotated CSV files
- Merges marked tissues and applies appropriate tissue type labels
- Creates labeled masks for downstream analysis
- Generates visualization of the final annotation

### 3. Region Suggestion (`suggest_regions.py`)
- Identifies regions of interest based on tissue type
- Applies edge removal to avoid boundary artifacts
- Generates windows of specified size for analysis
- Annotates sections with tissue metadata
- Outputs selected regions for further analysis

### 4. Pixel Clustering (`pixel_clustering.py`)
- Performs clustering of pixels to identify tissue components
- Uses dimensionality reduction (PCA) for visualization
- Applies clustering algorithms (K-means, GMM) for tissue segmentation
- Generates visualizations of clustering results

### 5. Vessel Detection (`mask_cnn_finetuning_vessels.py`)
- Fine-tunes a Mask R-CNN model on labeled vessel images
- Uses PyTorch's torchvision implementation of Mask R-CNN
- Trains on manually annotated vessel masks
- Generates predictions for vessel locations and shapes
- Outputs visualizations of detected vessels

## Selection Criteria

Sections are selected based on the following criteria:
- Sufficient distance from tissue edge
- Absence of blood vessels (or presence, depending on analysis goals)
- Appropriate tissue type (e.g., heart tissue)
- Minimal background

## Data Preparation

The pipeline includes tools for handling labeled data:
- `parse_exported_labels.py` - Processes exported label data from annotation tools
- Creates binary masks from polygon annotations
- Organizes images and masks for model training

## Model Training

The Mask R-CNN model is trained using:
- Manually annotated vessel images
- Binary masks indicating vessel locations
- PyTorch's implementation of Mask R-CNN with ResNet-50 backbone
- Fine-tuning approach to leverage pre-trained weights

## Output

The pipeline generates:
- Annotated images showing identified regions
- CSV files with region metadata
- Extracted image sections for further analysis
- Visualizations of the selection process
- Vessel detection masks and overlays
- Model training metrics and visualizations

## Usage

Each stage of the pipeline can be run independently, with outputs from earlier stages feeding into later ones. The typical workflow follows the numbered pipeline stages described above.

## Requirements

The pipeline requires several Python libraries:
- slideio - for handling slide images
- PyTorch - for deep learning models
- scikit-learn - for clustering and dimensionality reduction
- OpenCV - for image processing
- matplotlib - for visualization
- numpy/pandas - for data handling
