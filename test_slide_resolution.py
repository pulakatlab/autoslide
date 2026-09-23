"""
Script to load a slide and extract resolution information.
"""

import os
import sys
from autoslide.src.pipeline import utils
from autoslide.src import config

def main():
    # Get the SVS directory from config
    svs_dir = config['svs_dir']
    
    if not os.path.exists(svs_dir):
        print(f"SVS directory not found: {svs_dir}")
        print(f"Please ensure slides are in: {svs_dir}")
        return
    
    # Find first SVS file
    svs_files = [f for f in os.listdir(svs_dir) if f.endswith('.svs') or f.endswith('.vsi')]
    
    if not svs_files:
        print(f"No SVS or VSI files found in: {svs_dir}")
        return
    
    slide_path = os.path.join(svs_dir, svs_files[0])
    print(f"Loading slide: {slide_path}")
    
    try:
        # Load slide using the same method as suggest_regions.py
        slide_metadata = utils.slide_handler(slide_path, scene_index=0)
        scene = slide_metadata.scene
        
        # Extract resolution information
        resolution = scene.resolution[0]  # meters / pixel
        magnification = scene.magnification
        size = scene.size
        
        print(f"\n{'='*60}")
        print(f"Slide Resolution Information")
        print(f"{'='*60}")
        print(f"Slide file: {svs_files[0]}")
        print(f"Magnification: {magnification}x")
        print(f"Scene size (pixels): {size}")
        print(f"Native resolution: {resolution:.2e} meters/pixel")
        print(f"Native resolution: {resolution * 1e6:.6f} micrometers/pixel")
        print(f"Native resolution: {1 / (resolution * 1e6):.6f} pixels/micrometer")
        
        # Calculate ROI resolution (with 4x downsampling)
        roi_resolution_m_per_px = resolution * 4
        roi_resolution_um_per_px = roi_resolution_m_per_px * 1e6
        roi_resolution_px_per_um = 1 / roi_resolution_um_per_px
        
        print(f"\n{'='*60}")
        print(f"ROI Resolution (after 4x downsampling)")
        print(f"{'='*60}")
        print(f"ROI resolution: {roi_resolution_m_per_px:.2e} meters/pixel")
        print(f"ROI resolution: {roi_resolution_um_per_px:.6f} micrometers/pixel")
        print(f"ROI resolution: {roi_resolution_px_per_um:.6f} pixels/micrometer")
        
        # Calculate window size in pixels
        window_len = 7e-4  # meters
        window_shape_pixels = int(window_len / resolution)
        window_shape_pixels_downsampled = window_shape_pixels // 4
        
        print(f"\n{'='*60}")
        print(f"ROI Window Size")
        print(f"{'='*60}")
        print(f"Window size: {window_len * 1e6:.0f} micrometers")
        print(f"Window size (native pixels): {window_shape_pixels} x {window_shape_pixels}")
        print(f"Window size (downsampled pixels): {window_shape_pixels_downsampled} x {window_shape_pixels_downsampled}")
        
    except Exception as e:
        print(f"Error loading slide: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
