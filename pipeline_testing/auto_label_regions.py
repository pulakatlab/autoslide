"""
Auto-label regions for testing purposes.

This script automatically labels the largest thresholded region in the initial
annotation CSV as heart tissue to enable downstream processing without manual
annotation.

Usage:
    python pipeline_testing/auto_label_regions.py --data_dir <path>
    python pipeline_testing/auto_label_regions.py --csv_file <path_to_csv>
"""

import argparse
import os
import pandas as pd
import json


def auto_label_largest_region(csv_path, tissue_type="heart", tissue_num=1):
    """
    Label the largest region in the CSV as the specified tissue type.
    Label all other regions as 'other' to satisfy downstream requirements.
    
    Args:
        csv_path: Path to the initial annotation CSV file
        tissue_type: Tissue type label (default: "heart")
        tissue_num: Tissue number (default: 1)
    
    Returns:
        Path to the updated CSV file
    """
    print(f"Reading CSV: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} regions")
    print(f"Area range: {df['area'].min():.0f} - {df['area'].max():.0f}")
    
    # Find the largest region by area
    largest_idx = df['area'].idxmax()
    largest_area = df.loc[largest_idx, 'area']
    
    print(f"\nLargest region:")
    print(f"  Index: {largest_idx}")
    print(f"  Label: {df.loc[largest_idx, 'label']}")
    print(f"  Area: {largest_area:.0f}")
    
    # Label the largest region as the specified tissue type
    df.loc[largest_idx, 'tissue_type'] = tissue_type
    df.loc[largest_idx, 'tissue_num'] = tissue_num
    
    # Label all other regions as 'other' with tissue_num=2
    # This satisfies the requirement that all tissue_num values must be > 0
    other_indices = df.index[df.index != largest_idx]
    df.loc[other_indices, 'tissue_type'] = 'other'
    df.loc[other_indices, 'tissue_num'] = 2
    
    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Labeled largest region as '{tissue_type}' (tissue_num={tissue_num})")
    print(f"✓ Labeled {len(other_indices)} other region(s) as 'other' (tissue_num=2)")
    print(f"✓ Updated CSV saved to: {csv_path}")
    
    return csv_path


def process_data_directory(data_dir):
    """
    Process all CSV files in the initial_annotation directory.
    
    Args:
        data_dir: Base data directory containing initial_annotation subdirectory
    """
    init_annot_dir = os.path.join(data_dir, 'initial_annotation')
    
    if not os.path.exists(init_annot_dir):
        raise FileNotFoundError(f"Initial annotation directory not found: {init_annot_dir}")
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(init_annot_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {init_annot_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s) to process\n")
    
    for csv_file in csv_files:
        csv_path = os.path.join(init_annot_dir, csv_file)
        print(f"{'='*60}")
        print(f"Processing: {csv_file}")
        print(f"{'='*60}")
        auto_label_largest_region(csv_path)
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Auto-label regions for testing purposes'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_dir', type=str,
                       help='Base data directory containing initial_annotation subdirectory')
    group.add_argument('--csv_file', type=str,
                       help='Path to specific CSV file to process')
    
    parser.add_argument('--tissue_type', type=str, default='heart',
                        help='Tissue type label (default: heart)')
    parser.add_argument('--tissue_num', type=int, default=1,
                        help='Tissue number (default: 1)')
    
    args = parser.parse_args()
    
    if args.csv_file:
        # Process single CSV file
        if not os.path.exists(args.csv_file):
            raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
        auto_label_largest_region(args.csv_file, args.tissue_type, args.tissue_num)
    else:
        # Process all CSV files in data directory
        process_data_directory(args.data_dir)


if __name__ == '__main__':
    main()
