# Quick Start

Get up and running with AutoSlide in minutes.

## Running the Complete Pipeline

Execute the entire pipeline with a single command:

```bash
python src/pipeline/run_pipeline.py
```

This will run all pipeline stages in sequence:

1. Initial annotation
2. Final annotation
3. Region suggestion
4. Vessel detection
5. Fibrosis quantification

## Configuration

Before running the pipeline, set up your data directory in `src/config.json`:

```json
{
    "data_dir": "/path/to/your/data"
}
```

## Running Individual Pipeline Steps

You can execute specific stages independently:

### Initial Annotation

```bash
python src/pipeline/annotation/initial_annotation.py
```

### Final Annotation

```bash
python src/pipeline/annotation/final_annotation.py
```

### Region Suggestion

```bash
python src/pipeline/suggest_regions.py
```

### Vessel Detection

```bash
python src/pipeline/model/prediction.py
```

### Fibrosis Quantification

```bash
python src/fibrosis_calculation/calc_fibrosis.py
```

## Command Line Options

### Skip Annotation Steps

If you already have annotations:

```bash
python src/pipeline/run_pipeline.py --skip_annotation
```

### Fibrosis Quantification with Custom Parameters

```bash
python src/fibrosis_calculation/calc_fibrosis.py --hue-value 0.6785 --hue-width 0.4 --verbose
```

## Expected Output

AutoSlide generates comprehensive outputs including:

- **Annotated slide visualizations** with tissue boundaries and labels
- **Region selection maps** showing extracted analysis areas
- **Vessel detection results** with identified structures highlighted
- **Fibrosis quantification reports** with HSV-based percentage measurements and visualizations
- **Section tracking data** with unique identifiers for reproducibility
- **Quality control reports** from the mask validation GUI

## Next Steps

- Learn about [Configuration Options](configuration.md)
- Explore the [Pipeline Overview](../pipeline/overview.md)
- Use the [Mask Validation GUI](../tools/mask-validation-gui.md) for quality control
