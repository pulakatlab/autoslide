"""
Visualization module for autoslide.

This module provides functions for visualizing and comparing manual and automatic annotations.
"""

from .compare_positivity import (
    plot_positivity_comparison,
    plot_all_section_overlays,
    calculate_polygon_rectangle_overlap,
    find_matching_sections,
    analyze_overlap_threshold_sensitivity,
    create_overlap_threshold_analysis_plot,
)

from .overlay_manual_auto_annots import (
    plot_overlay_on_slide,
)

__all__ = [
    'plot_positivity_comparison',
    'plot_all_section_overlays',
    'calculate_polygon_rectangle_overlap',
    'find_matching_sections',
    'analyze_overlap_threshold_sensitivity',
    'create_overlap_threshold_analysis_plot',
    'plot_overlay_on_slide',
]
