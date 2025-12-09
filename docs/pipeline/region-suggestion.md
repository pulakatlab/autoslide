# Region Suggestion

The region suggestion stage strategically selects analysis-ready sections from annotated slides.

## Overview

This stage extracts optimal regions for vessel detection and fibrosis quantification based on tissue properties and analysis requirements.

## How It Works

1. **Load Annotations** - Import final annotation results
2. **Analyze Tissue Properties** - Evaluate tissue density, quality, and characteristics
3. **Score Regions** - Rank potential regions based on analysis criteria
4. **Extract Sections** - Cut out selected regions at appropriate resolution
5. **Generate Tracking IDs** - Create SHA-256 hashes for reproducibility
6. **Save Metadata** - Store region coordinates and properties

## Usage

```bash
python src/pipeline/suggest_regions.py
```

## Selection Criteria

Regions are selected based on:

- **Tissue density** - Sufficient tissue content
- **Quality metrics** - Minimal artifacts and staining issues
- **Size requirements** - Appropriate dimensions for analysis
- **Vessel presence** - Likelihood of containing vessels (for vessel detection)
- **Fibrotic characteristics** - Suitable for fibrosis quantification

## Unique Section Tracking

Each extracted region receives a unique SHA-256 hash based on:

- Source slide identifier
- Region coordinates
- Extraction parameters

This ensures reproducibility and prevents duplicate processing.

## Output

- Extracted region images (PNG format)
- Region tracking JSON with coordinates and metadata
- Visualization showing selected regions on original slide

## Next Steps

- [Vessel Detection](vessel-detection.md) - Detect vessels in extracted regions
- [Fibrosis Quantification](fibrosis-quantification.md) - Measure fibrosis in regions
