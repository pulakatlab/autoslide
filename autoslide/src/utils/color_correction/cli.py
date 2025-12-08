#!/usr/bin/env python3
"""
Command-line interface for color correction and normalization.

Usage examples:
    # Process all suggested_regions using config
    python -m autoslide.utils.color_correction.cli process-pipeline \\
        --reference-images data/reference/*.png \\
        --method reinhard \\
        --backup

    # Process specific SVS
    python -m autoslide.utils.color_correction.cli process-pipeline \\
        --reference-images data/reference/*.png \\
        --svs-name SVS_001 \\
        --backup

    # Process all images in a directory
    python -m autoslide.utils.color_correction.cli process-pipeline \\
        --input-dir data/my_images \\
        --reference-images data/reference/*.png \\
        --method reinhard \\
        --backup

    # Process images with backup
    python -m autoslide.utils.color_correction.cli process \\
        --input-dir data/images \\
        --reference-images data/reference/*.png \\
        --method reinhard \\
        --backup

    # Restore from backup
    python -m autoslide.utils.color_correction.cli restore \\
        --backup-dir data/images/backups/backup_20231126_120000

    # List available backups
    python -m autoslide.utils.color_correction.cli list-backups \\
        --backup-root data/images/backups
"""

import argparse
import logging
import sys
from pathlib import Path
from glob import glob

from autoslide.src.utils.color_correction.color_processor import (
    batch_process_directory,
    batch_process_suggested_regions,
    restore_originals,
    list_backups
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_process(args):
    """Process images with color correction/normalization."""
    # Handle reference images glob pattern
    if '*' in args.reference_images or '?' in args.reference_images:
        reference_images = glob(args.reference_images)
        if not reference_images:
            logger.error(
                f"No reference images found matching: {args.reference_images}")
            return 1
    else:
        reference_images = args.reference_images

    # Handle percentiles parameter
    percentiles = None
    if args.method == 'percentile_mapping':
        import numpy as np
        percentiles = np.linspace(0, 100, 101)

    result = batch_process_directory(
        input_dir=args.input_dir,
        reference_images=reference_images,
        method=args.method,
        percentiles=percentiles,
        file_pattern=args.file_pattern,
        backup=args.backup,
        backup_root=args.backup_root,
        replace_originals=args.replace_originals,
        output_dir=args.output_dir
    )

    if result['success']:
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Method: {result['method']}")
        print(f"Processed: {result['processed']} images")
        print(f"Failed: {result['failed']} images")
        print(f"Input directory: {result['input_dir']}")
        print(f"Output directory: {result['output_dir']}")

        if args.backup:
            print(f"Backup directory: {result['backup_dir']}")
            print(f"Backed up: {result['backup_count']} images")

        print("="*60)
        return 0
    else:
        logger.error(
            f"Processing failed: {result.get('error', 'Unknown error')}")
        return 1


def cmd_process_pipeline(args):
    """Process images in suggested_regions directory structure or a single input directory."""
    # Handle reference images glob pattern
    if '*' in args.reference_images or '?' in args.reference_images:
        reference_images = glob(args.reference_images)
        if not reference_images:
            logger.error(
                f"No reference images found matching: {args.reference_images}")
            return 1
    else:
        reference_images = args.reference_images

    # Handle percentiles parameter
    percentiles = None
    if args.method == 'percentile_mapping':
        import numpy as np
        percentiles = np.linspace(0, 100, 101)

    # If input_dir is provided, use batch_process_directory instead
    if args.input_dir:
        result = batch_process_directory(
            input_dir=args.input_dir,
            reference_images=reference_images,
            method=args.method,
            percentiles=percentiles,
            file_pattern=args.file_pattern,
            backup=args.backup,
            backup_root=args.backup_root,
            replace_originals=args.replace_originals,
            output_dir=args.output_dir
        )

        if result['success']:
            print("\n" + "="*60)
            print("DIRECTORY PROCESSING COMPLETE")
            print("="*60)
            print(f"Method: {result['method']}")
            print(f"Processed: {result['processed']} images")
            print(f"Failed: {result['failed']} images")
            print(f"Input directory: {result['input_dir']}")
            print(f"Output directory: {result['output_dir']}")

            if args.backup:
                print(f"Backup directory: {result['backup_dir']}")
                print(f"Backed up: {result['backup_count']} images")

            print("="*60)
            return 0
        else:
            logger.error(
                f"Processing failed: {result.get('error', 'Unknown error')}")
            return 1
    else:
        # Use the original suggested_regions processing
        result = batch_process_suggested_regions(
            reference_images=reference_images,
            method=args.method,
            percentiles=percentiles,
            file_pattern=args.file_pattern,
            backup=args.backup,
            replace_originals=args.replace_originals,
            svs_name=args.svs_name,
            suggested_regions_dir=args.suggested_regions_dir
        )

        if result['success']:
            print("\n" + "="*60)
            print("PIPELINE PROCESSING COMPLETE")
            print("="*60)
            print(f"SVS directories processed: {result['svs_count']}")
            print(f"Total images processed: {result['total_processed']}")
            print(f"Total failed: {result['total_failed']}")
            print("\nPer-SVS Results:")
            for svs_name, svs_result in result['svs_results'].items():
                print(f"\n  {svs_name}:")
                print(f"    Processed: {svs_result['processed']} images")
                print(f"    Failed: {svs_result['failed']} images")
                if args.backup:
                    print(f"    Backup: {svs_result['backup_dir']}")
            print("="*60)
            return 0
        else:
            logger.error(
                f"Processing failed: {result.get('error', 'Unknown error')}")
            return 1


def cmd_restore(args):
    """Restore original images from backup."""
    successful, failed = restore_originals(
        backup_dir=args.backup_dir,
        target_dir=args.target_dir,
        verify=not args.no_verify
    )

    print("\n" + "="*60)
    print("RESTORE COMPLETE")
    print("="*60)
    print(f"Restored: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Backup directory: {args.backup_dir}")
    if args.target_dir:
        print(f"Target directory: {args.target_dir}")
    print("="*60)

    return 0 if failed == 0 else 1


def cmd_list_backups(args):
    """List available backups."""
    backups = list_backups(args.backup_root)

    if not backups:
        print(f"No backups found in: {args.backup_root}")
        return 0

    print("\n" + "="*60)
    print("AVAILABLE BACKUPS")
    print("="*60)

    for i, backup in enumerate(backups, 1):
        print(f"\n{i}. Backup: {Path(backup['backup_dir']).name}")
        print(f"   Timestamp: {backup['timestamp']}")
        print(f"   Source: {backup['source_directory']}")
        print(f"   Files: {backup['file_count']}")
        print(f"   Method: {backup['method']}")
        print(f"   Path: {backup['backup_dir']}")

    print("\n" + "="*60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Color correction and normalization for histological images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Command to execute')
    subparsers.required = True

    # Process pipeline command (uses config)
    pipeline_parser = subparsers.add_parser(
        'process-pipeline',
        help='Process images in suggested_regions directory (uses autoslide config) or a single input directory'
    )
    pipeline_parser.add_argument(
        '--reference-images',
        required=True,
        help='Path to reference image(s) or glob pattern (e.g., "ref/*.png")'
    )
    pipeline_parser.add_argument(
        '--method',
        choices=['reinhard', 'histogram', 'percentile_mapping'],
        default='reinhard',
        help='Processing method (default: reinhard)'
    )
    pipeline_parser.add_argument(
        '--file-pattern',
        default='*.png',
        help='Glob pattern for image files (default: *.png)'
    )
    pipeline_parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup original images before processing'
    )
    pipeline_parser.add_argument(
        '--no-replace',
        dest='replace_originals',
        action='store_false',
        help='Save to output directory instead of replacing originals'
    )
    pipeline_parser.add_argument(
        '--input-dir',
        help='Process all images in this directory instead of using suggested_regions structure'
    )
    pipeline_parser.add_argument(
        '--backup-root',
        help='Root directory for backups when using --input-dir (default: input_dir/backups)'
    )
    pipeline_parser.add_argument(
        '--output-dir',
        help='Output directory when using --input-dir and --no-replace'
    )
    pipeline_parser.add_argument(
        '--svs-name',
        help='Process only specific SVS directory (default: all) - ignored when --input-dir is used'
    )
    pipeline_parser.add_argument(
        '--suggested-regions-dir',
        help='Override suggested_regions_dir from config - ignored when --input-dir is used'
    )
    pipeline_parser.set_defaults(func=cmd_process_pipeline)

    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Process images with color correction/normalization'
    )
    process_parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing images to process'
    )
    process_parser.add_argument(
        '--reference-images',
        required=True,
        help='Path to reference image(s) or glob pattern (e.g., "ref/*.png")'
    )
    process_parser.add_argument(
        '--method',
        choices=['reinhard', 'histogram', 'percentile_mapping'],
        default='reinhard',
        help='Processing method (default: reinhard)'
    )
    process_parser.add_argument(
        '--file-pattern',
        default='*.png',
        help='Glob pattern for image files (default: *.png)'
    )
    process_parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup original images before processing'
    )
    process_parser.add_argument(
        '--backup-root',
        help='Root directory for backups (default: input_dir/backups)'
    )
    process_parser.add_argument(
        '--no-replace',
        dest='replace_originals',
        action='store_false',
        help='Save to output directory instead of replacing originals'
    )
    process_parser.add_argument(
        '--output-dir',
        help='Output directory when not replacing originals'
    )
    process_parser.set_defaults(func=cmd_process)

    # Restore command
    restore_parser = subparsers.add_parser(
        'restore',
        help='Restore original images from backup'
    )
    restore_parser.add_argument(
        '--backup-dir',
        required=True,
        help='Directory containing backup images'
    )
    restore_parser.add_argument(
        '--target-dir',
        help='Directory to restore images to (default: original location from metadata)'
    )
    restore_parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification of backup metadata'
    )
    restore_parser.set_defaults(func=cmd_restore)

    # List backups command
    list_parser = subparsers.add_parser(
        'list-backups',
        help='List available backups'
    )
    list_parser.add_argument(
        '--backup-root',
        required=True,
        help='Root directory containing backup subdirectories'
    )
    list_parser.set_defaults(func=cmd_list_backups)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
