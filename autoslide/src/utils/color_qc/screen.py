"""
Pre-flight batch screening: given a directory of new slide-image ROI crops
(optionally grouped by slide, mirroring the ``suggested_regions/<SVS_NAME>/
images/`` convention already used elsewhere in this pipeline), scores each
slide -- and the batch overall -- against a reference color model and
reports whether the staining hue looks consistent with the training data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from autoslide.src.utils.color_qc.features import (
    FEATURE_NAMES, extract_color_features, aggregate_slide_features,
)
from autoslide.src.utils.color_qc.reference_model import ReferenceColorModel

DEFAULT_REFERENCE_MODEL_PATH = Path(
    __file__).parent / 'default_reference_model.json'
DEFAULT_FILE_PATTERN = '*.png'

_VERDICT_RANK = {
    'in_bounds': 0,
    'elevated': 1,
    'out_of_bounds': 2,
    'unknown (no thresholds calibrated)': -1,
}


def load_default_reference_model() -> ReferenceColorModel:
    """Load the reference model shipped with this package (fit from the 10
    known-good-acquisition training slides; see
    docs/advanced/color-qc-screening.md and
    autoslide_analysis/src/color_qc_validation.py for how it was produced
    and its validation limitations)."""
    return ReferenceColorModel.load(DEFAULT_REFERENCE_MODEL_PATH)


def _images_for_one_slide_dir(
    slide_dir: Path,
    file_pattern: str,
) -> List[Path]:
    """Resolve one slide's image list, preferring the ``images/`` subdirectory
    convention (``suggested_regions/<SVS_NAME>/images/*.png``, used
    everywhere else in this pipeline) over a flat layout, so a stray
    top-level file (e.g. a ``*_selected_section_visualization.png``, which
    real ``suggested_regions/<SVS_NAME>/`` directories contain) can't
    silently shadow the real ROI crops.
    """
    images_subdir = slide_dir / 'images'
    if images_subdir.is_dir():
        images = sorted(images_subdir.glob(file_pattern))
        if images:
            return images
    return sorted(slide_dir.glob(file_pattern))


def group_images_by_slide(
    input_dir: Union[str, Path],
    file_pattern: str = DEFAULT_FILE_PATTERN,
) -> Dict[str, List[Path]]:
    """Group images under ``input_dir`` by slide.

    If ``input_dir`` itself resolves to a non-empty image list (via
    ``_images_for_one_slide_dir`` -- ``images/`` subdirectory preferred,
    flat layout as a fallback), it is treated as one slide named after
    ``input_dir``. Otherwise, each subdirectory of ``input_dir`` is treated
    as one slide using the same resolution rule.
    """
    input_dir = Path(input_dir)
    direct_images = _images_for_one_slide_dir(input_dir, file_pattern)
    if direct_images:
        return {input_dir.name: direct_images}

    groups = {}
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        images = _images_for_one_slide_dir(child, file_pattern)
        if images:
            groups[child.name] = images
    return groups


def screen_slide(
    image_paths: List[Union[str, Path]],
    model: ReferenceColorModel,
) -> Dict[str, object]:
    """Screen one slide's ROI images against a reference model."""
    per_image = [extract_color_features(p) for p in image_paths]
    slide_features = aggregate_slide_features(per_image)
    result = model.score(slide_features)
    result['n_images'] = slide_features['n_images']
    result['n_images_used'] = slide_features['n_images_used']
    result['tissue_fraction'] = round(slide_features['tissue_fraction'], 4)
    result['_slide_features'] = slide_features
    return result


def screen_batch(
    input_dir: Union[str, Path],
    reference_model_path: Optional[Union[str, Path]] = None,
    file_pattern: str = DEFAULT_FILE_PATTERN,
) -> Dict[str, object]:
    """Screen a batch of new slide images against the reference color model.

    Returns a report dict with a per-slide verdict/distance, a batch-level
    summary (the batch centroid's own score), and the single worst-scoring
    slide -- non-fatal by design, the caller decides what to do with a
    flagged result (see ``format_report`` for a human-readable rendering,
    and ``run_pipeline.py``'s ``--check-color-qc`` flag for the pipeline
    integration, which only warns).
    """
    model = (ReferenceColorModel.load(reference_model_path) if reference_model_path
             else load_default_reference_model())

    groups = group_images_by_slide(input_dir, file_pattern)
    if not groups:
        raise ValueError(
            f"No images (pattern '{file_pattern}') found under {input_dir}")

    per_slide_results = {}
    all_slide_features = []
    for slide_name, image_paths in groups.items():
        result = screen_slide(image_paths, model)
        all_slide_features.append(result.pop('_slide_features'))
        per_slide_results[slide_name] = result

    batch_mean_features = {
        name: float(np.mean([sf[name] for sf in all_slide_features]))
        for name in FEATURE_NAMES
    }
    batch_result = model.score(batch_mean_features)

    worst_slide_name, worst_slide_result = max(
        per_slide_results.items(),
        key=lambda kv: _VERDICT_RANK.get(kv[1]['verdict'], -1),
    )

    return {
        'input_dir': str(input_dir),
        'n_slides': len(groups),
        'per_slide': per_slide_results,
        'batch_overall': batch_result,
        'worst_slide': {'name': worst_slide_name, **worst_slide_result},
    }


def format_report(report: Dict[str, object]) -> str:
    """Render a ``screen_batch`` report as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("COLOR/HUE PRE-FLIGHT SCREEN")
    lines.append("=" * 60)
    lines.append(f"Input: {report['input_dir']}")
    lines.append(f"Slides found: {report['n_slides']}")
    lines.append("")
    for slide_name, result in report['per_slide'].items():
        lines.append(
            f"  {slide_name}: verdict={result['verdict']} "
            f"d2={result['mahalanobis_sq']:.1f} "
            f"(n_images={result['n_images']}, "
            f"tissue_fraction={result['tissue_fraction']})"
        )
        if result['verdict'] != 'in_bounds':
            lines.append(
                f"    top contributing features: {result['top_feature_contributions']}")
    lines.append("")
    batch = report['batch_overall']
    lines.append(
        f"Batch overall: verdict={batch['verdict']} d2={batch['mahalanobis_sq']:.1f}")
    lines.append(
        f"Worst slide: {report['worst_slide']['name']} "
        f"(verdict={report['worst_slide']['verdict']})")
    lines.append("")
    if batch['verdict'] == 'in_bounds':
        lines.append(
            "This batch's staining hue looks consistent with the training data.")
    elif batch['verdict'] == 'elevated':
        lines.append(
            "This batch's staining hue is somewhat unusual relative to the "
            "training data -- recommend a visual spot-check of predictions "
            "before trusting them.")
    else:
        lines.append(
            "This batch's staining hue is well outside the training data's "
            "range -- do not trust predictions on it without revalidation.")
    lines.append("=" * 60)
    return "\n".join(lines)
