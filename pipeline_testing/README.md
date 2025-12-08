# AutoSlide Pipeline Testing

This directory contains testing infrastructure for the AutoSlide pipeline.

## Overview

The test pipeline orchestrates the main steps of the AutoSlide pipeline on test data:

1. **Initial Annotation** - Preliminary tissue delineation
2. **Final Annotation** - Tissue labeling and segmentation
3. **Region Suggestion** - Identify regions for detailed analysis
4. **Model Training** - Train vessel detection model
5. **Prediction** - Run predictions on selected regions

## Requirements

Install development dependencies (includes gdown for downloading test data):
```bash
pip install -r requirements-dev.txt
```

Install Git LFS and pull model files:
```bash
git lfs install
git lfs pull
```

This will download the trained models (~338MB total) from `autoslide/artifacts/`.

## Usage

### Download Test Data

Download test data from Google Drive:
```bash
python pipeline_testing/download_test_data.py
```

### Running Individual Pipeline Steps

After downloading test data, run each pipeline step directly:

```bash
# Initial annotation
python autoslide/src/pipeline/annotation/initial_annotation.py --verbose

# Auto-label regions for testing (bypasses manual annotation)
python pipeline_testing/auto_label_regions.py --data_dir test_data

# Final annotation
python autoslide/src/pipeline/annotation/final_annotation.py --verbose

# Region suggestion
python autoslide/src/pipeline/suggest_regions.py --verbose

# Prediction
python autoslide/src/pipeline/model/prediction.py --verbose

# Fibrosis calculation
python autoslide/src/fibrosis_calculation/calc_fibrosis.py --verbose
```

### Using Prefect for Local Testing (Optional)

For local development and testing, you can use the Prefect pipeline orchestrator:

```bash
# Run the full pipeline with downloaded test data
python pipeline_testing/prefect_pipeline.py --download_test_data --auto_label --verbose

# Or use your own test data
python pipeline_testing/prefect_pipeline.py --test_data path/to/file.svs --auto_label --verbose
```

**Note:** GitHub Actions workflows run scripts directly without Prefect for simpler execution and debugging.

### Prefect Pipeline Options (Local Testing Only)

When using `prefect_pipeline.py` for local testing:

- `--test_data PATH` - Path to test SVS file
- `--download_test_data` - Download test data from Google Drive
- `--auto_label` - Automatically label largest region as heart tissue (for testing)
- `--skip_annotation` - Skip annotation steps
- `--skip_training` - Skip model training step
- `--fail_fast` - Stop execution on first error
- `--verbose` - Enable verbose output

**Note:** Either `--test_data` or `--download_test_data` must be provided.

### Auto-Labeling for Testing

The auto-labeling script automatically labels the largest thresholded region as heart tissue, bypassing manual annotation for testing:

```bash
python pipeline_testing/auto_label_regions.py --data_dir test_data
```

### Examples

**Direct script execution (recommended for CI/CD):**
```bash
# Download and run complete pipeline
python pipeline_testing/download_test_data.py
python autoslide/src/pipeline/annotation/initial_annotation.py --verbose
python pipeline_testing/auto_label_regions.py --data_dir test_data
python autoslide/src/pipeline/annotation/final_annotation.py --verbose
python autoslide/src/pipeline/suggest_regions.py --verbose
python autoslide/src/pipeline/model/prediction.py --verbose
python autoslide/src/fibrosis_calculation/calc_fibrosis.py --verbose
```

**Prefect orchestration (local testing):**
```bash
# Download and run with auto-labeling
python pipeline_testing/prefect_pipeline.py --download_test_data --auto_label --verbose

# Use local test data
python pipeline_testing/prefect_pipeline.py \
    --test_data test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs \
    --auto_label \
    --verbose

# Skip training for faster testing
python pipeline_testing/prefect_pipeline.py --download_test_data --auto_label --skip_training --verbose
```

## Test Data

The pipeline can automatically download test data from Google Drive using the `--download_test_data` flag, or you can provide your own SVS file.

Downloaded test data is saved to:
```
<repo_root>/test_data/svs/
```

Test outputs are saved to:
```
<repo_root>/pipeline_testing/test_output/
```

Google Drive folder: [Test Data](https://drive.google.com/drive/folders/165Ei63lVEtCI1aQpKYIqhNR_zE8YeG5c)

## Pipeline Steps

The pipeline consists of the following steps:

1. **Download Test Data** - `download_test_data.py` - Downloads test SVS files from Google Drive
2. **Initial Annotation** - `initial_annotation.py` - Preliminary tissue delineation
3. **Auto-Label Regions** - `auto_label_regions.py` - Automatic labeling for testing (bypasses manual annotation)
4. **Final Annotation** - `final_annotation.py` - Tissue labeling and segmentation
5. **Region Suggestion** - `suggest_regions.py` - Identify regions for detailed analysis
6. **Prediction** - `prediction.py` - Run vessel detection on selected regions
7. **Fibrosis Calculation** - `calc_fibrosis.py` - Quantify fibrotic tissue

## CI/CD Integration

GitHub Actions workflows run each script directly without Prefect orchestration. This provides:
- Simpler execution and debugging
- Better visibility into individual step failures
- Faster execution without orchestration overhead
- More granular control over step execution

## Local Development with Prefect

For local testing, `prefect_pipeline.py` provides orchestration with:
- Automatic dependency management between steps
- Built-in monitoring and logging
- Error handling with `--fail_fast` option
- Progress tracking through Prefect UI (optional)

View execution details in console output or use the Prefect UI for detailed tracking.
