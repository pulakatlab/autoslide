#!/usr/bin/env python
"""
Verify the test logic for model availability checking.
"""

import os
import sys

def check_models():
    """Check which models are available"""
    artifacts_dir = os.path.join(
        os.path.dirname(__file__),
        'autoslide', 'artifacts'
    )
    
    models = {}
    
    # Check for Mask R-CNN model
    maskrcnn_path = os.path.join(artifacts_dir, 'best_val_mask_rcnn_model.pth')
    if os.path.exists(maskrcnn_path):
        models['maskrcnn'] = maskrcnn_path
        print(f"✅ Mask R-CNN model found: {maskrcnn_path}")
    else:
        print(f"❌ Mask R-CNN model not found: {maskrcnn_path}")
    
    # Check for UNet model
    unet_path = os.path.join(artifacts_dir, 'best_val_unet_model.pth')
    if os.path.exists(unet_path):
        models['unet'] = unet_path
        print(f"✅ UNet model found: {unet_path}")
    else:
        print(f"❌ UNet model not found: {unet_path}")
    
    return models

def main():
    print("Checking for available models...")
    print()
    
    models = check_models()
    
    print()
    print("Summary:")
    print(f"  Models found: {len(models)}")
    
    if not models:
        print("  ❌ FAIL: No models found. At least one model must be present.")
        return 1
    
    print(f"  ✅ PASS: {len(models)} model(s) available for testing")
    
    print()
    print("Test behavior:")
    for model_type in ['maskrcnn', 'unet']:
        if model_type in models:
            print(f"  - {model_type}: Will test (fail if prediction fails)")
        else:
            print(f"  - {model_type}: Will skip (model not available)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
