# 🔬 AutoSlide: AI-Powered Histological Analysis

AutoSlide is a comprehensive pipeline that transforms how researchers analyze histological slides, combining computer vision with deep learning to identify tissues, detect vessels, and quantify fibrosis.

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://pulakatlab.github.io/autoslide/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Automated Tissue Recognition** - Identify and classify tissue types
- **Smart Region Selection** - Extract informative regions for analysis
- **Vessel Detection** - Locate and measure blood vessels using Mask R-CNN
- **Fibrosis Quantification** - Measure fibrotic changes in tissue samples
- **Reproducible Workflow** - Consistent results with unique section tracking

## Quick Start

### Installation

```bash
git clone https://github.com/pulakatlab/autoslide.git
cd autoslide
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Configure data directory in src/config.json
python src/pipeline/run_pipeline.py
```

## Pipeline Stages

1. **Initial Annotation** - Tissue detection and region identification
2. **Final Annotation** - Precise tissue labeling and mask generation
3. **Region Suggestion** - Strategic section selection
4. **Vessel Detection** - Deep learning-based vessel identification
5. **Fibrosis Quantification** - HSV color-based tissue analysis

### Example Outputs

**Initial Annotation**
![Initial Annotation](https://github.com/user-attachments/assets/5e149cdc-6469-4fe7-9c11-4e710237eb35)

**Final Annotation**
<img src="https://github.com/user-attachments/assets/5976b0c1-0631-4360-8c65-9313ea431ffd" alt="Final Annotation" height="400">

**Region Suggestion**
<img src="https://github.com/user-attachments/assets/37600c55-e6da-4e7c-af2d-248f5ccdbb80" alt="Region Suggestion" height="400">

**Extracted Sections**
<img src="https://github.com/user-attachments/assets/315ffd0d-a0d8-4de3-b472-ae7cc939b65f" alt="Section 1" width="250">
<img src="https://github.com/user-attachments/assets/4399c97c-00d5-4efa-9612-a6806c8d1ac0" alt="Section 2" width="250">

## Documentation

For detailed documentation, visit [https://pulakatlab.github.io/autoslide/](https://pulakatlab.github.io/autoslide/)

- [Installation Guide](https://pulakatlab.github.io/autoslide/getting-started/installation/)
- [Quick Start Tutorial](https://pulakatlab.github.io/autoslide/getting-started/quickstart/)
- [Pipeline Overview](https://pulakatlab.github.io/autoslide/pipeline/overview/)
- [API Reference](https://pulakatlab.github.io/autoslide/api/pipeline/)
- [Mask Validation GUI](https://pulakatlab.github.io/autoslide/tools/mask-validation-gui/)

## Project Structure

```
autoslide/
├── src/
│   ├── pipeline/           # Main pipeline modules
│   ├── utils/             # Utilities and GUI tools
│   └── fibrosis_calculation/  # Fibrosis analysis
├── docs/                  # Documentation source
├── tests/                 # Test suite
└── requirements.txt       # Dependencies
```

## Contributing

Contributions are welcome! See our [Contributing Guide](https://pulakatlab.github.io/autoslide/contributing/) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
