"""
Pre-flight color/hue screening: characterizes the training data's Masson's
Trichrome staining hue/color as a reference envelope, and scores new
batches against it so a drifted batch can be flagged before its
predictions are trusted.
"""

from .features import (
    FEATURE_NAMES,
    extract_color_features,
    aggregate_slide_features,
)
from .reference_model import ReferenceColorModel
from .screen import (
    screen_batch,
    screen_slide,
    group_images_by_slide,
    load_default_reference_model,
    format_report,
)

__all__ = [
    'FEATURE_NAMES',
    'extract_color_features',
    'aggregate_slide_features',
    'ReferenceColorModel',
    'screen_batch',
    'screen_slide',
    'group_images_by_slide',
    'load_default_reference_model',
    'format_report',
]
