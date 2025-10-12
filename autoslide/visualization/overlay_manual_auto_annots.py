"""
Plot overlay of manual and automatic annotations on svs image
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import ast

from autoslide.pipeline.utils import slide_handler


def plot_overlay_on_slide(
        slide_path, 
        manual_annotations, 
        auto_annotations, 
        output_path=None, 
        downsample=100
        ):
    """
    Plot annotations on slide image.
    
    Args:
        slide_path (str): Path to SVS slide file
        manual_annotations (pd.DataFrame): DataFrame of manual annotations for the slide
        auto_annotations (pd.DataFrame): DataFrame of automatic annotations for the slide
        output_path (str): Path to save output image (optional)
        downsample (int): Downsample factor for display
    """
    print(f"  Loading slide: {os.path.basename(slide_path)}")
    # Load slide
    try:
        handler = slide_handler(slide_path)
        scene = handler.scene
    except Exception as e:
        print(f"  Error loading slide {slide_path}: {e}")
        return

    # Get downsampled image
    print(f"  Creating downsampled image (factor: {downsample})")
    image_shape = scene.rect[2:]
    down_shape = [int(i/downsample) for i in image_shape]
    small_image = scene.read_block(size=down_shape)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(small_image)
    
    # Calculate scaling factor
    scale_x = small_image.shape[1] / image_shape[0]
    scale_y = small_image.shape[0] / image_shape[1]
    
    # Plot manual annotations
    print(f"  Plotting {len(manual_annotations)} manual annotations")
    for _, ann in manual_annotations.iterrows():
        try:
            vertices_x = ast.literal_eval(ann['vertices_x'])
            vertices_y = ast.literal_eval(ann['vertices_y'])
            vertices = list(zip(vertices_x, vertices_y))
            
            if len(vertices) >= 4:
                scaled_vertices = [(int(x * scale_x), int(y * scale_y)) for x, y in vertices]
                polygon = patches.Polygon(scaled_vertices, linewidth=1, 
                                        edgecolor='orange', facecolor='orange', alpha=0.4)
                ax.add_patch(polygon)
        except Exception as e:
            print(f"Could not plot manual annotation: {ann}, error: {e}")
    
    # Plot automatic annotations
    print(f"  Plotting {len(auto_annotations)} automatic annotations")
    for _, ann in auto_annotations.iterrows():
        try:
            bounds = ast.literal_eval(ann['section_bounds'])
            x_min, y_min, x_max, y_max = bounds
            
            # Scale to downsampled image
            x = x_min * scale_x
            y = y_min * scale_y
            width = (x_max - x_min) * scale_x
            height = (y_max - y_min) * scale_y

            # Create rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                     edgecolor='cyan', facecolor='cyan', alpha=0.4)
            ax.add_patch(rect)
        except Exception as e:
            print(f"Could not plot automatic annotation: {ann}, error: {e}")

    ax.set_title(f"Annotations for {os.path.basename(slide_path)}")
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved annotated image to: {output_path}")
    else:
        plt.show()
    
    plt.close()
