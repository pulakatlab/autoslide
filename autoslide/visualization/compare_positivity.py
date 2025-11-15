"""
For manual and automated annotations, compare the positivity scores
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import ast
import argparse
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from autoslide.pipeline.utils import slide_handler


def plot_positivity_comparison(
        slide_path, 
        manual_annotations, 
        auto_annotations, 
        output_path=None, 
        downsample=100
        ):
    """
    Plot side-by-side comparison of manual and automatic annotations with positivity-based coloring.
    
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
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display image on both subplots
    ax1.imshow(small_image)
    ax2.imshow(small_image)
    
    # Calculate scaling factor
    scale_x = small_image.shape[1] / image_shape[0]
    scale_y = small_image.shape[0] / image_shape[1]
    
    # Collect all positivity scores to determine data range
    all_manual_pos = []
    all_auto_pos = []
    
    for _, ann in manual_annotations.iterrows():
        try:
            positivity = float(ann['Positivity_positive_ratio'])
            all_manual_pos.append(positivity)
        except:
            pass
    
    for _, ann in auto_annotations.iterrows():
        try:
            positivity = float(ann['fibrosis_percentage']) / 100
            all_auto_pos.append(positivity)
        except:
            pass
    
    # Determine the maximum positivity score in current data
    max_positivity = max(max(all_manual_pos) if all_manual_pos else 0, 
                        max(all_auto_pos) if all_auto_pos else 0)
    
    # Define colormap for positivity scores
    cmap = plt.cm.plasma
    
    # Plot manual annotations on left subplot
    print(f"  Plotting {len(manual_annotations)} manual annotations")
    manual_positivity_scores = []
    
    for _, ann in manual_annotations.iterrows():
        try:
            vertices_x = ast.literal_eval(ann['vertices_x'])
            vertices_y = ast.literal_eval(ann['vertices_y'])
            vertices = list(zip(vertices_x, vertices_y))
            
            # Get positivity score
            positivity = float(ann['Positivity_positive_ratio'])
            manual_positivity_scores.append(positivity)
            
            if len(vertices) >= 4:
                scaled_vertices = [(int(x * scale_x), int(y * scale_y)) for x, y in vertices]
                
                # Color based on positivity score, normalized to max in data
                color = cmap(positivity / max_positivity if max_positivity > 0 else 0)
                
                polygon = patches.Polygon(scaled_vertices, linewidth=1, 
                                        edgecolor='black', facecolor=color, alpha=1.0)
                ax1.add_patch(polygon)
                
        except Exception as e:
            print(f"Could not plot manual annotation: {e}")
    
    # Plot automatic annotations on right subplot
    print(f"  Plotting {len(auto_annotations)} automatic annotations")
    auto_positivity_scores = []
    
    for _, ann in auto_annotations.iterrows():
        try:
            bounds = ast.literal_eval(ann['section_bounds'])
            x_min, y_min, x_max, y_max = bounds
            
            # Get positivity score
            positivity = float(ann['fibrosis_percentage']) / 100
            auto_positivity_scores.append(positivity)
            
            # Scale to downsampled image
            x = x_min * scale_x
            y = y_min * scale_y
            width = (x_max - x_min) * scale_x
            height = (y_max - y_min) * scale_y

            # Color based on positivity score, normalized to max in data
            color = cmap(positivity / max_positivity if max_positivity > 0 else 0)
            
            # Create rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                   edgecolor='black', facecolor=color, alpha=1.0)
            ax2.add_patch(rect)
            
        except Exception as e:
            print(f"Could not plot automatic annotation: {e}")

    # Set titles and formatting
    ax1.set_title(f'Manual Annotations\n{os.path.basename(slide_path)}\n'
                  f'Mean Positivity: {np.mean(manual_positivity_scores):.3f} ± {np.std(manual_positivity_scores):.3f}', 
                  fontsize=12)
    ax2.set_title(f'Automatic Annotations\n{os.path.basename(slide_path)}\n'
                  f'Mean Positivity: {np.mean(auto_positivity_scores):.3f} ± {np.std(auto_positivity_scores):.3f}', 
                  fontsize=12)
    
    ax1.axis('off')
    ax2.axis('off')
    
    # Add colorbar scaled to actual data range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_positivity))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', fraction=0.05)
    cbar.set_label(f'Positivity Score (max: {max_positivity:.3f})', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return manual_positivity_scores, auto_positivity_scores


def plot_all_section_overlays(slide_path, matched_pairs, output_path):
    """
    Plot overlay of all matched manual and automatic section pairs for one slide.
    
    Args:
        slide_path (str): Path to SVS slide file
        matched_pairs (list): List of tuples (manual_ann, auto_ann, overlap_score)
        output_path (str): Path to save output image
    """
    if not matched_pairs:
        return
        
    try:
        # Load slide
        handler = slide_handler(slide_path)
        scene = handler.scene
        
        # Get downsampled full image
        downsample = 100
        image_shape = scene.rect[2:]
        down_shape = [int(i/downsample) for i in image_shape]
        full_image = scene.read_block(size=down_shape)
        
        # Calculate scaling factor
        scale_x = full_image.shape[1] / image_shape[0]
        scale_y = full_image.shape[0] / image_shape[1]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(full_image)
        
        # Define consistent colors for manual and automatic annotations
        manual_color = 'orange'
        auto_color = 'cyan'
        
        # Plot all matched pairs
        for i, (manual_ann, auto_ann, overlap_score) in enumerate(matched_pairs, 1):
            # Get manual annotation vertices
            vertices_x = ast.literal_eval(manual_ann['vertices_x'])
            vertices_y = ast.literal_eval(manual_ann['vertices_y'])
            manual_vertices = list(zip(vertices_x, vertices_y))
            
            # Get auto annotation bounds
            auto_bounds = ast.literal_eval(auto_ann['section_bounds'])
            x_min_auto, y_min_auto, x_max_auto, y_max_auto = auto_bounds
            
            # Plot manual annotation (scale to downsampled image)
            scaled_vertices = [(int(x * scale_x), int(y * scale_y)) for x, y in manual_vertices]
            manual_positivity = float(manual_ann['Positivity_positive_ratio'])
            
            polygon = patches.Polygon(scaled_vertices, linewidth=2, 
                                    edgecolor=manual_color, facecolor='none', alpha=0.8,
                                    linestyle='-')
            ax.add_patch(polygon)
            
            # Plot automatic annotation (scale to downsampled image)
            auto_x = x_min_auto * scale_x
            auto_y = y_min_auto * scale_y
            auto_width = (x_max_auto - x_min_auto) * scale_x
            auto_height = (y_max_auto - y_min_auto) * scale_y
            auto_positivity = float(auto_ann['fibrosis_percentage']) / 100
            
            rect = patches.Rectangle((auto_x, auto_y), auto_width, auto_height, 
                                   linewidth=2, edgecolor=auto_color, facecolor='none', alpha=0.8,
                                   linestyle='--')
            ax.add_patch(rect)
            
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color=manual_color, linewidth=2, linestyle='-', label='Manual Annotations'),
            plt.Line2D([0], [0], color=auto_color, linewidth=2, linestyle='--', label='Automatic Annotations')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f"All Section Overlays - {os.path.basename(slide_path)}\n"
                    f"Total Matches: {len(matched_pairs)}")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating section overlay plot: {e}")


def calculate_polygon_rectangle_overlap(polygon_vertices, rectangle_bounds):
    """
    Calculate the overlap area between a polygon and a rectangle.
    
    Args:
        polygon_vertices (list): List of (x, y) tuples for polygon vertices
        rectangle_bounds (tuple): (x_min, y_min, x_max, y_max) for rectangle
        
    Returns:
        float: Overlap area as fraction of polygon area
    """
    try:
        # Create shapely geometries
        polygon = Polygon(polygon_vertices)
        x_min, y_min, x_max, y_max = rectangle_bounds
        rectangle = box(x_min, y_min, x_max, y_max)
        
        # Calculate intersection
        intersection = polygon.intersection(rectangle)
        
        # Return overlap fraction relative to polygon area
        if polygon.area > 0:
            return intersection.area / polygon.area
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error calculating overlap: {e}")
        return 0.0


def find_matching_sections(manual_annotations, auto_annotations, overlap_threshold=0.5, 
                          slide_path=None, output_dir=None, slide_name=None):
    """
    Find matching manual and automatic sections based on area overlap.
    
    Args:
        manual_annotations (pd.DataFrame): Manual annotations for a slide
        auto_annotations (pd.DataFrame): Automatic annotations for a slide
        overlap_threshold (float): Minimum overlap fraction to consider a match
        slide_path (str): Path to SVS slide file for overlay plotting
        output_dir (str): Directory to save overlay plots
        slide_name (str): Name of slide for output filenames
        
    Returns:
        tuple: (matches, matched_pairs) where matches is list of (manual_positivity, auto_positivity) 
               and matched_pairs is list of (manual_ann, auto_ann, overlap_score) for plotting
    """
    matches = []
    matched_pairs = []
    
    for _, manual_ann in manual_annotations.iterrows():
        try:
            # Get manual annotation polygon
            vertices_x = ast.literal_eval(manual_ann['vertices_x'])
            vertices_y = ast.literal_eval(manual_ann['vertices_y'])
            manual_vertices = list(zip(vertices_x, vertices_y))
            manual_positivity = float(manual_ann['Positivity_positive_ratio'])
            
            best_overlap = 0
            best_auto_positivity = None
            best_auto_ann = None
            
            # Check overlap with all automatic annotations
            for _, auto_ann in auto_annotations.iterrows():
                try:
                    auto_bounds = ast.literal_eval(auto_ann['section_bounds'])
                    auto_positivity = float(auto_ann['fibrosis_percentage']) / 100
                    
                    # Calculate overlap
                    overlap = calculate_polygon_rectangle_overlap(manual_vertices, auto_bounds)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_auto_positivity = auto_positivity
                        best_auto_ann = auto_ann
                        
                except Exception as e:
                    continue
            
            # If best overlap meets threshold, add to matches
            if best_overlap >= overlap_threshold and best_auto_positivity is not None:
                matches.append((manual_positivity, best_auto_positivity))
                matched_pairs.append((manual_ann, best_auto_ann, best_overlap))
                
        except Exception as e:
            continue
    
    # Create single overlay plot for all matched pairs if slide_path is provided
    if matched_pairs and slide_path and output_dir and slide_name:
        overlay_output = os.path.join(output_dir, f"{slide_name}_all_overlays.png")
        plot_all_section_overlays(slide_path, matched_pairs, overlay_output)
    
    return matches, matched_pairs


def analyze_overlap_threshold_sensitivity(manual_annotations, auto_annotations, 
                                        thresholds=None, slide_name=None):
    """
    Analyze how correlation changes with different overlap thresholds.
    
    Args:
        manual_annotations (pd.DataFrame): Manual annotations for a slide
        auto_annotations (pd.DataFrame): Automatic annotations for a slide
        thresholds (list): List of overlap thresholds to test
        slide_name (str): Name of slide for identification
        
    Returns:
        dict: Dictionary with threshold as key and matches data as value
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.05, 0.1)
    
    threshold_results = {}
    
    for threshold in thresholds:
        matches, _ = find_matching_sections(
            manual_annotations, auto_annotations, 
            overlap_threshold=threshold
        )
        
        threshold_results[threshold] = {
            'matches': matches,
            'n_matches': len(matches),
            'slide_name': slide_name
        }
    
    return threshold_results


def create_overlap_threshold_analysis_plot(all_threshold_results, output_path=None):
    """
    Create plots showing how correlation metrics change with overlap threshold.
    Pool all ROI matches across slides for each threshold before calculating correlation.
    
    Args:
        all_threshold_results (list): List of threshold_results dictionaries from all slides
        output_path (str): Path to save output image (optional)
    """
    if not all_threshold_results:
        print("No threshold results found for analysis plot")
        return
    
    # Combine results from all slides
    thresholds = sorted(list(all_threshold_results[0].keys()))
    
    # Pool matches across all slides for each threshold
    threshold_metrics = {}
    for threshold in thresholds:
        all_matches = []
        n_slides_with_matches = 0
        
        # Collect all matches across slides for this threshold
        for slide_results in all_threshold_results:
            if threshold in slide_results:
                slide_matches = slide_results[threshold]['matches']
                all_matches.extend(slide_matches)
                if len(slide_matches) > 0:
                    n_slides_with_matches += 1
        
        # Calculate correlation using pooled matches
        if len(all_matches) >= 2:
            manual_scores = [match[0] for match in all_matches]
            auto_scores = [match[1] for match in all_matches]
            
            correlation, p_value = stats.pearsonr(manual_scores, auto_scores)
            
            # Calculate R-squared
            X = np.array(manual_scores).reshape(-1, 1)
            y = np.array(auto_scores)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            
            threshold_metrics[threshold] = {
                'correlation': correlation,
                'p_value': p_value,
                'r2': r2,
                'total_matches': len(all_matches),
                'n_slides_with_matches': n_slides_with_matches,
                'manual_mean': np.mean(manual_scores),
                'auto_mean': np.mean(auto_scores),
                'manual_std': np.std(manual_scores),
                'auto_std': np.std(auto_scores)
            }
        else:
            threshold_metrics[threshold] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'r2': np.nan,
                'total_matches': len(all_matches),
                'n_slides_with_matches': n_slides_with_matches,
                'manual_mean': np.nan,
                'auto_mean': np.nan,
                'manual_std': np.nan,
                'auto_std': np.nan
            }
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    threshold_vals = list(threshold_metrics.keys())
    correlations = [threshold_metrics[t]['correlation'] for t in threshold_vals]
    r2_vals = [threshold_metrics[t]['r2'] for t in threshold_vals]
    p_values = [threshold_metrics[t]['p_value'] for t in threshold_vals]
    n_matches = [threshold_metrics[t]['total_matches'] for t in threshold_vals]
    
    # Plot 1: Correlation vs Threshold
    ax1.plot(threshold_vals, correlations, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Overlap Threshold', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Correlation vs Overlap Threshold\n(Pooled ROIs)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: R-squared vs Threshold
    ax2.plot(threshold_vals, r2_vals, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Overlap Threshold', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² vs Overlap Threshold\n(Pooled ROIs)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of matches vs Threshold
    ax3.plot(threshold_vals, n_matches, 'o-', color='red', linewidth=2, markersize=8)
    ax3.set_xlabel('Overlap Threshold', fontsize=12)
    ax3.set_ylabel('Total Number of Matches', fontsize=12)
    ax3.set_title('Total Matches vs Overlap Threshold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: P-value vs Threshold (log scale)
    valid_p_vals = [(t, p) for t, p in zip(threshold_vals, p_values) if not np.isnan(p)]
    if valid_p_vals:
        t_vals, p_vals = zip(*valid_p_vals)
        ax4.semilogy(t_vals, p_vals, 'o-', color='purple', linewidth=2, markersize=8)
        ax4.axhline(y=0.05, color='r', linestyle='--', label='p=0.05', alpha=0.5)
        ax4.axhline(y=0.01, color='r', linestyle=':', label='p=0.01', alpha=0.5)
        ax4.legend()
    ax4.set_xlabel('Overlap Threshold', fontsize=12)
    ax4.set_ylabel('P-value (log scale)', fontsize=12)
    ax4.set_title('Statistical Significance vs Overlap Threshold', fontsize=14)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved threshold analysis plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return threshold_metrics
