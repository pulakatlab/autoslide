# Automated Slide Analysis Pipeline

This project provides a pipeline for automated analysis of histological slides, particularly focusing on tissue identification, region extraction, and fibrosis analysis.

## Overview

The pipeline processes histological slides (SVS format) through several stages:
1. Initial thresholding to extract tissue samples
2. Region identification and annotation
3. Extraction of sections with specific properties
4. Analysis of selected regions

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

## Selection Criteria

Sections are selected based on the following criteria:
- Sufficient distance from tissue edge
- Absence of blood vessels
- Appropriate tissue type (e.g., heart tissue)
- Minimal background

## Output

The pipeline generates:
- Annotated images showing identified regions
- CSV files with region metadata
- Extracted image sections for further analysis
- Visualizations of the selection process

## Usage

Each stage of the pipeline can be run independently, with outputs from earlier stages feeding into later ones. The typical workflow follows the numbered pipeline stages described above.
