"""
Visualize automatic annotations from the pipeline with positivity-based coloring
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import ast

from autoslide.pipeline.utils import slide_handler


def plot_annotations_with_positivity(
        slide_path, 
        auto_annotations, 
        output_path=None, 
        downsample=100
        ):
    """
    Plot automatic annotations with positivity-based coloring.
    
    Args:
        slide_path (str): Path to SVS slide file
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
    
    # Collect all positivity scores to determine data range
    all_auto_pos = []
    
    for _, ann in auto_annotations.iterrows():
        try:
            positivity = float(ann['fibrosis_percentage']) / 100
            all_auto_pos.append(positivity)
        except:
            pass
    
    # Determine the maximum positivity score in current data
    max_positivity = max(all_auto_pos) if all_auto_pos else 1.0
    
    # Define colormap for positivity scores
    cmap = plt.cm.plasma
    
    # Plot automatic annotations
    print(f"  Plotting {len(auto_annotations)} automatic annotations")
    auto_positivity_scores = []
    
    for _, ann in auto_annotations.iterrows():
        try:
            bounds = ast.literal_eval(ann['section_bounds'])
            x_min, y_min, x_max, y_max = bounds
            
            # Get positivity score
            positivity = float(ann['fibrosis_percentage']) / 100
            auto_positivity_scores.append(positivity)
            
            # Normalize positivity to [0, 1] for colormap
            norm_positivity = positivity / max_positivity if max_positivity > 0 else 0
            color = cmap(norm_positivity)
            
            # Scale to downsampled image
            x = x_min * scale_x
            y = y_min * scale_y
            width = (x_max - x_min) * scale_x
            height = (y_max - y_min) * scale_y

            # Create rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                     edgecolor=color, facecolor=color, alpha=0.5)
            ax.add_patch(rect)
        except Exception as e:
            print(f"Could not plot automatic annotation: {ann}, error: {e}")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_positivity))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fibrosis Percentage', rotation=270, labelpad=20)
    
    # Add statistics
    if auto_positivity_scores:
        stats_text = f"Auto Annotations: n={len(auto_positivity_scores)}, "
        stats_text += f"mean={np.mean(auto_positivity_scores):.3f}, "
        stats_text += f"std={np.std(auto_positivity_scores):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f"Automatic Annotations for {os.path.basename(slide_path)}")
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved annotated image to: {output_path}")
    else:
        plt.show()
    
    plt.close()
