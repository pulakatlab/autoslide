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

Install Prefect:
```bash
pip install prefect>=2.0.0
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the full pipeline on test data:
```bash
python pipeline_testing/prefect_pipeline.py --test_data test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs
```

### Options

- `--test_data PATH` - Path to test SVS file (required)
- `--skip_annotation` - Skip annotation steps
- `--skip_training` - Skip model training step
- `--fail_fast` - Stop execution on first error
- `--verbose` - Enable verbose output

### Examples

Skip annotation steps:
```bash
python pipeline_testing/prefect_pipeline.py \
    --test_data test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs \
    --skip_annotation
```

Run with verbose output and fail-fast:
```bash
python pipeline_testing/prefect_pipeline.py \
    --test_data test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs \
    --verbose \
    --fail_fast
```

## Test Data

The pipeline expects test data in SVS format. The default test data path is:
```
<repo_root>/test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs
```

Test outputs are saved to:
```
<repo_root>/pipeline_testing/test_output/
```

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
