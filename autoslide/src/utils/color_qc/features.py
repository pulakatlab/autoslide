"""
Per-image and per-slide color/hue feature extraction for the color_qc
pre-flight screening module.

Feature values are computed on tissue-only pixels (a simple saturation
threshold, matching the approach already used by
``autoslide.src.fibrosis_calculation.calc_fibrosis``) and are intended to
characterize Masson's Trichrome staining hue/color, not general image
content.
"""

from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np

# Same hue band autoslide's fibrosis colorimetric threshold uses for
# collagen (blue) detection -- reused rather than re-derived. Source:
# calc_fibrosis.py's default fibrosis_config, itself citing
# https://pmc.ncbi.nlm.nih.gov/articles/PMC5376943/. Hue is normalized to
# [0, 1] here (OpenCV reports uint8 hue in [0, 180]).
COLLAGEN_HUE_VALUE = 0.6785
COLLAGEN_HUE_WIDTH = 0.4

# Muscle (red) band, symmetric around hue 0/1 (hue wraps at the ends of the
# [0, 1] range). Chosen as an initial default matching the empirically
# observed muscle-hue cluster on real training images; see
# autoslide_analysis/src/color_qc_validation.py for the real-data check.
MUSCLE_HUE_WIDTH = 0.2

TISSUE_SATURATION_THRESHOLD = 0.08
LOG_RATIO_EPSILON = 1e-3

# Model-input feature names, in the fixed order ReferenceColorModel uses.
FEATURE_NAMES = [
    'lab_mean_l', 'lab_mean_a', 'lab_mean_b',
    'lab_std_a', 'lab_std_b',
    'hsv_mean_saturation', 'hsv_mean_value',
    'log_blue_red_ratio',
]


def _load_image_rgb(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image as RGB uint8 (H, W, 3), accepting a path or an array.

    A path is read via cv2.imread (BGR) and converted to RGB, matching the
    convention calc_fibrosis.gen_fibrosis_mask expects for its own hue-band
    computation. An array input is assumed already RGB.
    """
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"Expected an HxWx3(+) image array, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _tissue_mask(
    hsv_image: np.ndarray,
    saturation_threshold: float = TISSUE_SATURATION_THRESHOLD,
) -> np.ndarray:
    """Boolean mask of tissue (non-background/glass) pixels via a saturation
    threshold -- mirrors the lightweight approach calc_fibrosis.py already
    uses. These ROI crops are pre-selected to be tissue-dense (by
    suggest_regions.py), so a heavier whole-slide tissue detector is not
    warranted here.
    """
    saturation = hsv_image[:, :, 1].astype(np.float32) / 255.0
    return saturation >= saturation_threshold


def extract_color_features(
    image: Union[str, Path, np.ndarray],
    saturation_threshold: float = TISSUE_SATURATION_THRESHOLD,
) -> Dict[str, float]:
    """Extract a compact color/hue feature vector from one Trichrome ROI image.

    Features characterize overall staining hue/intensity (LAB mean/std,
    HSV saturation/value) plus a Trichrome-specific collagen:muscle
    log-ratio, computed on tissue-only pixels (background/glass excluded
    via a saturation threshold). See ``FEATURE_NAMES`` for the keys used as
    model inputs; ``tissue_fraction`` and ``other_hue_fraction`` are
    returned as separate diagnostic signals, not model inputs -- the log
    ratio's fraction inputs are compositional, so they should not also be
    fed into a covariance model alongside their own ratio.

    Returns all-NaN model-input features (with a real ``tissue_fraction``)
    if no tissue pixels are detected at all, rather than silently computing
    statistics over background/glass pixels.
    """
    rgb = _load_image_rgb(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    tissue = _tissue_mask(hsv, saturation_threshold)
    n_pixels = rgb.shape[0] * rgb.shape[1]
    tissue_fraction = float(tissue.sum()) / n_pixels if n_pixels else 0.0

    if not tissue.any():
        nan_features = {name: float('nan') for name in FEATURE_NAMES}
        nan_features['tissue_fraction'] = tissue_fraction
        nan_features['other_hue_fraction'] = float('nan')
        return nan_features

    lab_tissue = lab[tissue]
    hsv_tissue = hsv[tissue].astype(np.float32)

    hue_norm = hsv_tissue[:, 0] / 180.0  # OpenCV uint8 hue is in [0, 180]
    saturation_norm = hsv_tissue[:, 1] / 255.0
    value_norm = hsv_tissue[:, 2] / 255.0

    collagen_low = COLLAGEN_HUE_VALUE - COLLAGEN_HUE_WIDTH / 2
    collagen_high = COLLAGEN_HUE_VALUE + COLLAGEN_HUE_WIDTH / 2
    is_collagen = (hue_norm >= collagen_low) & (hue_norm <= collagen_high)

    # Muscle band wraps around 0/1 (hue is circular): [0, w/2] U [1-w/2, 1]
    is_muscle = (hue_norm <= MUSCLE_HUE_WIDTH / 2) | (
        hue_norm >= 1 - MUSCLE_HUE_WIDTH / 2)

    n_tissue = hue_norm.shape[0]
    f_blue = float(is_collagen.sum()) / n_tissue
    f_red = float(is_muscle.sum()) / n_tissue
    f_other = max(0.0, 1.0 - f_blue - f_red)

    log_ratio = float(
        np.log((f_blue + LOG_RATIO_EPSILON) / (f_red + LOG_RATIO_EPSILON)))

    return {
        'lab_mean_l': float(lab_tissue[:, 0].mean()),
        'lab_mean_a': float(lab_tissue[:, 1].mean()),
        'lab_mean_b': float(lab_tissue[:, 2].mean()),
        'lab_std_a': float(lab_tissue[:, 1].std()),
        'lab_std_b': float(lab_tissue[:, 2].std()),
        'hsv_mean_saturation': float(saturation_norm.mean()),
        'hsv_mean_value': float(value_norm.mean()),
        'log_blue_red_ratio': log_ratio,
        'tissue_fraction': tissue_fraction,
        'other_hue_fraction': f_other,
    }


def aggregate_slide_features(
    per_image_features: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate a slide's per-ROI-image features to per-slide summary
    statistics -- the correct independent unit for fitting/scoring against
    the reference model (ROI crops from the same slide are correlated, not
    independent draws; see docs/advanced/color-qc-screening.md).

    Returns the mean of each ``FEATURE_NAMES`` entry (the model-input
    vector for this slide), a ``<feature>_within_slide_std`` entry per
    feature (within-slide spread -- a diagnostic, not a model input), the
    mean ``tissue_fraction``/``other_hue_fraction`` diagnostics, and
    ``n_images``/``n_images_used``.
    """
    if not per_image_features:
        raise ValueError("per_image_features must be non-empty")

    valid = [f for f in per_image_features
             if not np.isnan(f['log_blue_red_ratio'])]
    if not valid:
        raise ValueError(
            "No per-image features had any detected tissue pixels")

    result = {
        'n_images': len(per_image_features),
        'n_images_used': len(valid),
    }
    for name in FEATURE_NAMES:
        values = np.array([f[name] for f in valid], dtype=np.float64)
        result[name] = float(values.mean())
        result[f'{name}_within_slide_std'] = float(values.std())
    result['tissue_fraction'] = float(
        np.mean([f['tissue_fraction'] for f in valid]))
    result['other_hue_fraction'] = float(
        np.mean([f['other_hue_fraction'] for f in valid]))
    return result
