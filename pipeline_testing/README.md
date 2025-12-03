# AutoSlide Pipeline Testing

This directory contains testing infrastructure for the AutoSlide pipeline using Prefect.

## Overview

The Prefect test pipeline (`prefect_pipeline.py`) orchestrates the main steps of the AutoSlide pipeline on test data:

1. **Initial Annotation** - Preliminary tissue delineation
2. **Final Annotation** - Tissue labeling and segmentation
3. **Region Suggestion** - Identify regions for detailed analysis
4. **Model Training** - Train vessel detection model
5. **Prediction** - Run predictions on selected regions

## Requirements

Install development dependencies (includes Prefect and gdown):
```bash
pip install -r requirements-dev.txt
```

## Usage

### Download Test Data

Download test data from Google Drive:
```bash
python pipeline_testing/prefect_pipeline.py --download_test_data
```

### Basic Usage

Run the full pipeline with downloaded test data:
```bash
python pipeline_testing/prefect_pipeline.py --download_test_data
```

Or use your own test data:
```bash
python pipeline_testing/prefect_pipeline.py --test_data path/to/file.svs
```

### Options

- `--test_data PATH` - Path to test SVS file
- `--download_test_data` - Download test data from Google Drive
- `--auto_label` - Automatically label largest region as heart tissue (for testing)
- `--skip_annotation` - Skip annotation steps
- `--skip_training` - Skip model training step
- `--fail_fast` - Stop execution on first error
- `--verbose` - Enable verbose output

**Note:** Either `--test_data` or `--download_test_data` must be provided.

### Auto-Labeling for Testing

The `--auto_label` flag enables automatic labeling of the largest thresholded region as heart tissue. This bypasses the manual annotation step and allows downstream processing to continue for testing purposes.

```bash
python pipeline_testing/prefect_pipeline.py --download_test_data --auto_label --verbose
```

### Examples

Download and run with auto-labeling (recommended for testing):
```bash
python pipeline_testing/prefect_pipeline.py --download_test_data --auto_label --verbose
```

Skip annotation steps with downloaded data:
```bash
python pipeline_testing/prefect_pipeline.py --download_test_data --skip_annotation
```

Use local test data with auto-labeling:
```bash
python pipeline_testing/prefect_pipeline.py \
    --test_data test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs \
    --auto_label \
    --verbose
```

Manual labeling (without auto-label):
```bash
# Run initial annotation
python pipeline_testing/prefect_pipeline.py --download_test_data --skip_annotation

# Manually edit the CSV file in test_data/initial_annotation/
# Then run remaining steps
python pipeline_testing/prefect_pipeline.py --test_data <path> --verbose
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

Each step is implemented as a Prefect task:

- `verify_test_data()` - Validates test data file exists
- `setup_test_environment()` - Creates output directories
- `run_initial_annotation()` - Executes initial annotation
- `run_final_annotation()` - Executes final annotation
- `run_region_suggestion()` - Executes region suggestion
- `run_model_training()` - Executes model training
- `run_prediction()` - Executes prediction

## Error Handling

By default, the pipeline continues on errors. Use `--fail_fast` to stop on the first error encountered.

## Monitoring

Prefect provides built-in monitoring and logging. View execution details in the console output or use the Prefect UI for more detailed tracking.
