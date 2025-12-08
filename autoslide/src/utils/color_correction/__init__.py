"""
Unified color correction module for histological images.

This module provides color correction and normalization with backup/restore functionality.
"""

from .color_processor import (
    ColorProcessor,
    batch_process_directory,
    batch_process_suggested_regions,
    find_svs_image_directories,
    restore_originals,
    list_backups
)

__all__ = [
    'ColorProcessor',
    'batch_process_directory',
    'batch_process_suggested_regions',
    'find_svs_image_directories',
    'restore_originals',
    'list_backups'
]
