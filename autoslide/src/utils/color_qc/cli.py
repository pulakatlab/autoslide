#!/usr/bin/env python3
"""
Command-line interface for the color_qc pre-flight screening tool.

Usage examples:
    # Screen a batch that mirrors the suggested_regions/<SVS_NAME>/images/
    # convention (each subdirectory is treated as one slide)
    python -m autoslide.src.utils.color_qc.cli screen \\
        --input-dir data/new_batch

    # Screen a flat directory of images (treated as a single slide)
    python -m autoslide.src.utils.color_qc.cli screen \\
        --input-dir data/new_batch/images

    # Screen against a custom (re-fit) reference model instead of the
    # shipped default
    python -m autoslide.src.utils.color_qc.cli screen \\
        --input-dir data/new_batch \\
        --reference-model path/to/reference_model.json \\
        --json-output report.json
"""

import argparse
import json
import logging
import sys

from autoslide.src.utils.color_qc.screen import screen_batch, format_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_screen(args):
    """Screen a batch of new slide images against the reference color model."""
    try:
        report = screen_batch(
            input_dir=args.input_dir,
            reference_model_path=args.reference_model,
            file_pattern=args.file_pattern,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    print(format_report(report))

    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull JSON report written to: {args.json_output}")

    return 0 if report['batch_overall']['verdict'] != 'out_of_bounds' else 1


def main():
    parser = argparse.ArgumentParser(
        description='Pre-flight color/hue screening for AutoSlide slide batches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Command to execute')
    subparsers.required = True

    screen_parser = subparsers.add_parser(
        'screen',
        help='Screen a batch of new slide images against the reference color model'
    )
    screen_parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory of new slide images -- either a flat directory of '
             'images (one slide) or a directory of per-slide subdirectories '
             '(each containing images/ or images directly)'
    )
    screen_parser.add_argument(
        '--reference-model',
        default=None,
        help='Path to a reference_model.json (default: the model shipped '
             'with this package, fit from the 10 known-good training slides)'
    )
    screen_parser.add_argument(
        '--file-pattern',
        default='*.png',
        help='Glob pattern for image files (default: *.png)'
    )
    screen_parser.add_argument(
        '--json-output',
        default=None,
        help='Optional path to also write the full report as JSON'
    )
    screen_parser.set_defaults(func=cmd_screen)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
