# GitHub Workflows

## test-pipeline.yml

Automated testing workflow for the AutoSlide pipeline.

### Triggers

- Push to `main` or `test_merge_prs_88_86` branches
- Pull requests to `main`
- Manual trigger via workflow_dispatch

### What It Does

1. **Setup Environment**
   - Ubuntu latest
   - Python 3.12
   - System dependencies (OpenGL, GLib)
   - Python dependencies from requirements.txt and requirements-dev.txt

2. **Create Mock Model**
   - Generates a test Mask R-CNN model for prediction testing

3. **Download Test Data**
   - Downloads SVS test file from Google Drive (~528MB)

4. **Run Pipeline Steps**
   - Initial annotation (tissue detection)
   - Auto-labeling (labels largest region as heart)
   - Final annotation (processes labeled regions)
   - Region suggestion (extracts heart tissue sections)
   - Prediction (generates vessel masks)
   - Fibrosis calculation (quantifies fibrosis)

5. **Generate Test Summary**
   - Creates markdown summary of results
   - Reports success/failure of each step
   - Includes statistics from fibrosis calculation

### Artifacts Generated

The workflow uploads multiple artifacts (retained for 30 days):

1. **test-summary** - Markdown summary of test results
2. **initial-annotation** - Initial tissue masks and metadata
3. **final-annotation** - Final processed masks
4. **region-suggestions** - Heart tissue section images and visualizations
5. **prediction-results** - Vessel detection masks and overlays
6. **fibrosis-results** - Fibrosis quantification results and plots
7. **complete-test-data** - All test outputs (7 days retention)

### Viewing Results

After workflow completion:
1. Go to Actions tab in GitHub
2. Click on the workflow run
3. Scroll to "Artifacts" section
4. Download desired artifact zip files

### Expected Outputs

- **Initial annotation**: 3 files (PNG, CSV, NPY)
- **Final annotation**: 2 files (PNG, NPY)
- **Region suggestions**: 14 heart tissue section images
- **Prediction**: 14 masks + 14 overlays
- **Fibrosis results**: CSV, JSON, summary plots, 14 visualizations

### Timeout

Workflow has 30-minute timeout to prevent excessive resource usage.

### Notes

- Uses `continue-on-error: true` for pipeline steps to ensure artifacts are uploaded even if steps fail
- Excludes large files (SVS, NPY) from complete-test-data artifact
- Mock model is used for prediction (not a trained model)
