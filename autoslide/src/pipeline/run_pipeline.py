#!/usr/bin/env python
# coding: utf-8

"""
AutoSlide: Complete Pipeline Runner

This script orchestrates the complete AutoSlide pipeline:
1. Initial annotation of slide images
2. Final annotation with tissue labels
3. Region suggestion for detailed analysis
4. Model training for vessel detection
5. Prediction on selected regions

Usage:
    python run_pipeline.py [--data_dir PATH] [--skip_annotation] [--skip_training]
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f"autoslide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutoSlide")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoSlide Pipeline Runner")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (overrides default in scripts)')
    parser.add_argument('--skip_annotation', action='store_true',
                        help='Skip initial and final annotation steps')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable dataset augmentation during training')
    parser.add_argument('--check_color_qc', action='store_true',
                        help='Screen the suggested-regions batch for staining '
                             'hue/color consistent with the training data before '
                             'running prediction. Non-fatal: a flagged batch only '
                             'logs a warning, the pipeline still runs.')
    return parser.parse_args()


def run_initial_annotation(data_dir=None):
    """Run the initial annotation step"""
    logger.info("Starting initial annotation...")
    from autoslide.src.pipeline.annotation.initial_annotation import main as initial_annotation_main

    if data_dir:
        # Set data directory if provided
        pass

    try:
        initial_annotation_main()
        logger.info("Initial annotation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in initial annotation: {str(e)}")
        return False


def run_final_annotation(data_dir=None):
    """Run the final annotation step"""
    logger.info("Starting final annotation...")
    from autoslide.src.pipeline.annotation.final_annotation import main as final_annotation_main

    try:
        final_annotation_main()
        logger.info("Final annotation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in final annotation: {str(e)}")
        return False


def run_region_suggestion(data_dir=None):
    """Run the region suggestion step"""
    logger.info("Starting region suggestion...")
    from autoslide.src.pipeline.suggest_regions import main as suggest_regions_main

    try:
        suggest_regions_main()
        logger.info("Region suggestion completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in region suggestion: {str(e)}")
        return False


def run_model_training(data_dir=None, use_augmentation=True):
    """Run the model training step"""
    logger.info("Starting model training...")
    if use_augmentation:
        logger.info(
            "Using dataset augmentation with negative samples and artificial vessels")
    from autoslide.src.pipeline.model.training import main as training_main

    try:
        # Note: The training.py script now handles augmentation internally
        training_main()
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return False


def run_color_qc_screen(data_dir=None):
    """Screen the suggested-regions batch for staining hue/color consistent
    with the training data. Non-fatal by design (see
    docs/advanced/color-qc-screening.md): a flagged batch only logs a
    warning -- the caller decides whether to stop and revalidate."""
    logger.info("Starting color/hue pre-flight screen...")
    from autoslide.src import config
    from autoslide.src.utils.color_qc.screen import screen_batch, format_report

    try:
        suggested_regions_dir = config['suggested_regions_dir']
        report = screen_batch(suggested_regions_dir)
        logger.info("\n" + format_report(report))
        if report['batch_overall']['verdict'] != 'in_bounds':
            logger.warning(
                f"Color/hue pre-flight screen flagged this batch as "
                f"'{report['batch_overall']['verdict']}' -- staining hue looks "
                f"unusual relative to the training data. Predictions on this "
                f"batch may be less reliable; consider a visual spot-check. "
                f"(Pipeline continues regardless.)"
            )
        return True
    except Exception as e:
        logger.error(f"Error in color/hue pre-flight screen: {str(e)}")
        return False


def run_prediction(data_dir=None):
    """Run the prediction step"""
    logger.info("Starting prediction...")
    from autoslide.src.pipeline.model.prediction import main as prediction_main

    try:
        prediction_main()
        logger.info("Prediction completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return False


def main():
    """Main pipeline runner"""
    args = parse_args()

    logger.info("Starting AutoSlide pipeline...")

    # Run each step of the pipeline
    if not args.skip_annotation:
        if not run_initial_annotation(args.data_dir):
            logger.warning(
                "Initial annotation failed, but continuing pipeline...")

        if not run_final_annotation(args.data_dir):
            logger.warning(
                "Final annotation failed, but continuing pipeline...")
    else:
        logger.info("Skipping annotation steps as requested")

    if not run_region_suggestion(args.data_dir):
        logger.warning("Region suggestion failed, but continuing pipeline...")

    if not args.skip_training:
        if not run_model_training(args.data_dir, use_augmentation=not args.no_augmentation):
            logger.warning("Model training failed, but continuing pipeline...")
    else:
        logger.info("Skipping model training as requested")

    if args.check_color_qc:
        if not run_color_qc_screen(args.data_dir):
            logger.warning(
                "Color/hue pre-flight screen failed to run, but continuing pipeline...")

    if not run_prediction(args.data_dir):
        logger.warning("Prediction failed")

    logger.info("AutoSlide pipeline completed")


if __name__ == "__main__":
    main()
