"""
Resources
- https://github.com/DatumLearning/Mask-RCNN-finetuning-PyTorch/blob/main/notebook.ipynb
- https://www.youtube.com/watch?v=vV9L71hK-RE
- https://www.youtube.com/watch?v=t1MrzuAUdoE

"""

#!/usr/bin/env python
# coding: utf-8


from autoslide.pipeline.model.training_utils import (
    setup_directories, setup_training, train_model, plot_losses, evaluate_model, load_model
)
from autoslide.pipeline.model.prediction_utils import initialize_model
from autoslide.pipeline.model.data_preprocessing import (
    prepare_data, plot_augmented_samples, create_sample_plots
)
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import cv2
import torch
import argparse

# Import config
from autoslide import config

# Get directories from config
data_dir = config['data_dir']
artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
plot_dir = config['plot_dirs']

# Import utilities directly

##############################
##############################


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Mask R-CNN model for vessel detection')
    parser.add_argument('--retrain', action='store_true', 
                       help='Force retraining even if a saved model exists')
    parser.add_argument('--n-runs', type=int, default=1,
                       help='Number of training runs to perform (default: 1)')
    return parser.parse_args()


def main():
    """Main function to run the training pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    plot_dir, artifacts_dir = setup_directories(data_dir)

    # Prepare all data using the preprocessing pipeline
    data_components = prepare_data(data_dir, use_augmentation=True)
    
    # Extract components
    train_dl = data_components['train_dl']
    val_dl = data_components['val_dl']
    train_imgs = data_components['train_imgs']
    train_masks = data_components['train_masks']
    val_imgs = data_components['val_imgs']
    val_masks = data_components['val_masks']
    img_dir = data_components['img_dir']
    mask_dir = data_components['mask_dir']
    aug_img_dir = data_components['aug_img_dir']
    aug_mask_dir = data_components['aug_mask_dir']
    aug_img_names = data_components['aug_img_names']
    aug_mask_names = data_components['aug_mask_names']

    # Create visualization plots
    if len(aug_img_names) > 0:
        plot_augmented_samples(aug_img_dir, aug_mask_dir,
                               aug_img_names, aug_mask_names, plot_dir)

    create_sample_plots(
        train_imgs, train_masks, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        plot_dir
    )

    # Initialize model
    model = initialize_model()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Setup training
    optimizer = setup_training(model, device)

    # Model paths
    best_model_path = os.path.join(
        artifacts_dir, 'best_val_mask_rcnn_model.pth')

    # Load existing model or train new one
    if os.path.exists(best_model_path) and not args.retrain:
        print('Loading model from savefile')
        model = load_model(model, best_model_path, device)
    else:
        # Perform multiple training runs if requested
        if args.n_runs > 1:
            print(f'\nPerforming {args.n_runs} training runs to select best model...')
            best_overall_val_loss = float('inf')
            best_run_idx = -1
            run_results = []
            
            for run_idx in range(args.n_runs):
                print(f'\n{"="*60}')
                print(f'Starting training run {run_idx + 1}/{args.n_runs}')
                print(f'{"="*60}\n')
                
                # Reinitialize model and optimizer for each run
                run_model = initialize_model()
                run_optimizer = setup_training(run_model, device)
                
                # Train model for this run
                run_model, run_train_losses, run_val_losses, run_best_val_loss = train_model(
                    run_model, train_dl, val_dl, run_optimizer, device, plot_dir, artifacts_dir,
                    run_idx=run_idx
                )
                
                # Store results
                run_results.append({
                    'run_idx': run_idx,
                    'model_state': run_model.state_dict(),
                    'train_losses': run_train_losses,
                    'val_losses': run_val_losses,
                    'best_val_loss': run_best_val_loss
                })
                
                print(f'\nRun {run_idx + 1} completed with best validation loss: {run_best_val_loss:.4f}')
                
                # Track best run
                if run_best_val_loss < best_overall_val_loss:
                    best_overall_val_loss = run_best_val_loss
                    best_run_idx = run_idx
                    print(f'*** New best model found in run {run_idx + 1} ***')
            
            # Select and save the best model
            print(f'\n{"="*60}')
            print(f'Training complete! Best model from run {best_run_idx + 1}')
            print(f'Best validation loss: {best_overall_val_loss:.4f}')
            print(f'{"="*60}\n')
            
            # Load the best model
            best_result = run_results[best_run_idx]
            model.load_state_dict(best_result['model_state'])
            
            # Save the best model
            torch.save(best_result['model_state'], best_model_path)
            
            # Save all run results for analysis
            run_summary_path = os.path.join(artifacts_dir, 'training_runs_summary.npy')
            np.save(run_summary_path, {
                'best_run_idx': best_run_idx,
                'best_val_loss': best_overall_val_loss,
                'all_runs': [{
                    'run_idx': r['run_idx'],
                    'best_val_loss': r['best_val_loss'],
                    'train_losses': r['train_losses'],
                    'val_losses': r['val_losses']
                } for r in run_results]
            })
            
            # Plot comparison of all runs
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            for r in run_results:
                label = f"Run {r['run_idx'] + 1}"
                if r['run_idx'] == best_run_idx:
                    label += " (Best)"
                ax1.plot(r['train_losses'], label=label, 
                        linewidth=2 if r['run_idx'] == best_run_idx else 1)
                ax2.plot(r['val_losses'], label=label,
                        linewidth=2 if r['run_idx'] == best_run_idx else 1)
            
            ax1.set_title('Training Loss - All Runs')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('Validation Loss - All Runs')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'all_runs_comparison.png'), dpi=150)
            plt.close()
            
            print(f'Run comparison plot saved to {plot_dir}/all_runs_comparison.png')
            
        else:
            # Single training run (original behavior)
            model, all_train_losses, all_val_losses, best_val_loss = train_model(
                model, train_dl, val_dl, optimizer, device, plot_dir, artifacts_dir
            )

    # Evaluate model
    evaluate_model(
        model, val_imgs, val_masks,
        img_dir, mask_dir, aug_img_dir, aug_mask_dir,
        device, plot_dir
    )


if __name__ == "__main__":
    main()
