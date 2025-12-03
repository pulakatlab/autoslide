"""
Prefect pipeline for testing AutoSlide main steps on test data

This pipeline runs the main steps of the AutoSlide pipeline:
1. Initial annotation
2. Final annotation  
3. Region suggestion
4. Model training
5. Prediction

Usage:
    python pipeline_testing/prefect_pipeline.py --test_data <path_to_svs_file>
    python pipeline_testing/prefect_pipeline.py --test_data test_data/svs/x_8142-2021_Trichrome_426_427_37727.svs
"""

import argparse
import os
import sys
from subprocess import PIPE, Popen
from pathlib import Path

from prefect import flow, task
import gdown

# Add project root to path
script_path = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run AutoSlide pipeline tests on test data')
    parser.add_argument('--test_data', type=str, required=False,
                        help='Path to test SVS file')
    parser.add_argument('--download_test_data', action='store_true',
                        help='Download test data from Google Drive')
    parser.add_argument('--skip_annotation', action='store_true',
                        help='Skip annotation steps')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--fail_fast', action='store_true',
                        help='Stop execution on first error')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def raise_error_if_error(process, stderr, stdout, fail_fast=True, verbose=False):
    """Check process return code and handle errors"""
    if verbose or process.returncode:
        print('=== Process stdout ===\n')
        print(stdout.decode('utf-8'))
        print('\n=== Process stderr ===\n')
        print(stderr.decode('utf-8'))
    
    if process.returncode:
        decode_err = stderr.decode('utf-8')
        if fail_fast:
            raise Exception(f"Process failed with error:\n{decode_err}")
        else:
            print('Encountered error...fail-fast not enabled, continuing execution...\n')


@task(log_prints=True)
def download_test_data():
    """Download test data from Google Drive"""
    print("Downloading test data from Google Drive...")
    
    # Google Drive folder ID
    folder_id = "165Ei63lVEtCI1aQpKYIqhNR_zE8YeG5c"
    
    # Create test data directory
    test_data_dir = os.path.join(project_root, 'test_data', 'svs')
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Download folder contents
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading from: {folder_url}")
    
    try:
        gdown.download_folder(url=folder_url, output=test_data_dir, quiet=False, use_cookies=False)
        print(f"Test data downloaded to: {test_data_dir}")
        
        # Find the first SVS file (recursively search subdirectories)
        svs_files = []
        for root, dirs, files in os.walk(test_data_dir):
            for file in files:
                if file.endswith('.svs'):
                    svs_files.append(os.path.join(root, file))
        
        if not svs_files:
            raise FileNotFoundError("No SVS files found in downloaded data")
        
        test_file = svs_files[0]
        print(f"Using test file: {test_file}")
        return test_file
        
    except Exception as e:
        print(f"Error downloading test data: {e}")
        raise


@task(log_prints=True)
def verify_test_data(test_data_path):
    """Verify test data file exists"""
    print(f"Verifying test data at: {test_data_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    if not test_data_path.endswith('.svs'):
        raise ValueError(f"Test data must be an SVS file, got: {test_data_path}")
    
    print(f"Test data verified: {test_data_path}")
    return test_data_path


@task(log_prints=True)
def setup_test_environment(test_data_path):
    """Setup test environment and data directory"""
    print("Setting up test environment...")
    
    # Create test output directory
    test_output_dir = os.path.join(project_root, 'pipeline_testing', 'test_output')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy test data to expected location if needed
    test_data_dir = os.path.join(project_root, 'test_data', 'svs')
    os.makedirs(test_data_dir, exist_ok=True)
    
    print(f"Test output directory: {test_output_dir}")
    print(f"Test data directory: {test_data_dir}")
    
    return test_output_dir


@task(log_prints=True)
def run_initial_annotation(test_data_path, fail_fast=True, verbose=False):
    """Run initial annotation step"""
    print("Running initial annotation...")
    script_name = os.path.join(project_root, 'autoslide', 'src', 'pipeline', 
                               'annotation', 'initial_annotation.py')
    
    cmd = ["python", script_name]
    if verbose:
        cmd.append('--verbose')
    
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout, fail_fast, verbose)
    print("Initial annotation completed")


@task(log_prints=True)
def run_final_annotation(test_data_path, fail_fast=True, verbose=False):
    """Run final annotation step"""
    print("Running final annotation...")
    script_name = os.path.join(project_root, 'autoslide', 'src', 'pipeline',
                               'annotation', 'final_annotation.py')
    
    cmd = ["python", script_name]
    if verbose:
        cmd.append('--verbose')
    
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout, fail_fast, verbose)
    print("Final annotation completed")


@task(log_prints=True)
def run_region_suggestion(test_data_path, fail_fast=True, verbose=False):
    """Run region suggestion step"""
    print("Running region suggestion...")
    script_name = os.path.join(project_root, 'autoslide', 'src', 'pipeline',
                               'suggest_regions.py')
    
    cmd = ["python", script_name]
    if verbose:
        cmd.append('--verbose')
    
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout, fail_fast, verbose)
    print("Region suggestion completed")


@task(log_prints=True)
def run_model_training(test_data_path, fail_fast=True, verbose=False):
    """Run model training step"""
    print("Running model training...")
    script_name = os.path.join(project_root, 'autoslide', 'src', 'pipeline',
                               'model', 'training.py')
    
    cmd = ["python", script_name]
    if verbose:
        cmd.append('--verbose')
    
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout, fail_fast, verbose)
    print("Model training completed")


@task(log_prints=True)
def run_prediction(test_data_path, fail_fast=True, verbose=False):
    """Run prediction step"""
    print("Running prediction...")
    script_name = os.path.join(project_root, 'autoslide', 'src', 'pipeline',
                               'model', 'prediction.py')
    
    cmd = ["python", script_name]
    if verbose:
        cmd.append('--verbose')
    
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    raise_error_if_error(process, stderr, stdout, fail_fast, verbose)
    print("Prediction completed")


@flow(name="autoslide-test-pipeline", log_prints=True)
def autoslide_test_pipeline(
    test_data_path=None,
    download_data=False,
    skip_annotation=False,
    skip_training=False,
    fail_fast=True,
    verbose=False
):
    """
    Main Prefect flow for testing AutoSlide pipeline
    
    Args:
        test_data_path: Path to test SVS file (optional if download_data=True)
        download_data: Download test data from Google Drive
        skip_annotation: Skip annotation steps
        skip_training: Skip model training
        fail_fast: Stop on first error
        verbose: Enable verbose output
    """
    print("=" * 60)
    print("AutoSlide Test Pipeline")
    print("=" * 60)
    
    # Download or verify test data
    if download_data:
        verified_path = download_test_data()
    elif test_data_path:
        verified_path = verify_test_data(test_data_path)
    else:
        raise ValueError("Either --test_data or --download_test_data must be provided")
    
    # Setup environment
    test_output_dir = setup_test_environment(verified_path)
    
    # Run pipeline steps
    if not skip_annotation:
        run_initial_annotation(verified_path, fail_fast, verbose)
        run_final_annotation(verified_path, fail_fast, verbose)
    else:
        print("Skipping annotation steps")
    
    run_region_suggestion(verified_path, fail_fast, verbose)
    
    if not skip_training:
        run_model_training(verified_path, fail_fast, verbose)
    else:
        print("Skipping model training")
    
    run_prediction(verified_path, fail_fast, verbose)
    
    print("=" * 60)
    print("AutoSlide Test Pipeline Completed Successfully")
    print("=" * 60)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate arguments
    if not args.download_test_data and not args.test_data:
        raise ValueError("Either --test_data or --download_test_data must be provided")
    
    # Convert relative path to absolute if provided
    test_data_path = os.path.abspath(args.test_data) if args.test_data else None
    
    # Run the flow
    autoslide_test_pipeline(
        test_data_path=test_data_path,
        download_data=args.download_test_data,
        skip_annotation=args.skip_annotation,
        skip_training=args.skip_training,
        fail_fast=args.fail_fast,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
