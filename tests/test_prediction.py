"""
Tests for prediction functionality with both Mask R-CNN and UNet models.

Test logic:
- If both models are present, test both
- If a model is present but fails, fail the test
- If a model isn't present, skip it
- If no models are found, fail the test
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from autoslide.src.pipeline.model.prediction_utils import (
    load_model, predict_single_image, initialize_model
)


@pytest.fixture
def sample_test_image(tmp_path):
    """Create a sample test image"""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.png"
    Image.fromarray(img).save(img_path)
    return str(img_path)


@pytest.fixture
def available_models():
    """
    Check which models are available for testing.
    
    Returns:
        dict: Dictionary with model types as keys and paths as values
    """
    artifacts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'autoslide', 'artifacts'
    )
    
    models = {}
    
    # Check for Mask R-CNN model
    maskrcnn_path = os.path.join(artifacts_dir, 'best_val_mask_rcnn_model.pth')
    if os.path.exists(maskrcnn_path):
        models['maskrcnn'] = maskrcnn_path
    
    # Check for UNet model
    unet_path = os.path.join(artifacts_dir, 'best_val_unet_model.pth')
    if os.path.exists(unet_path):
        models['unet'] = unet_path
    
    return models


class TestModelInitialization:
    """Test model initialization for both architectures"""
    
    def test_initialize_maskrcnn(self):
        """Test Mask R-CNN model initialization"""
        model = initialize_model(model_type='maskrcnn')
        
        assert hasattr(model, 'roi_heads')
        assert hasattr(model, 'backbone')
        assert model.roi_heads.box_predictor.cls_score.out_features == 2
        assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
    
    def test_initialize_unet(self):
        """Test UNet model initialization"""
        model = initialize_model(model_type='unet')
        
        assert hasattr(model, 'inc')
        assert hasattr(model, 'down1')
        assert hasattr(model, 'up1')
        assert hasattr(model, 'outc')
        assert model.n_channels == 3
        assert model.n_classes == 1
    
    def test_initialize_invalid_model(self):
        """Test that invalid model type raises error"""
        with pytest.raises(ValueError, match="Unknown model type"):
            initialize_model(model_type='invalid_model')


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_maskrcnn_if_available(self, available_models):
        """Test loading Mask R-CNN model if available"""
        if 'maskrcnn' not in available_models:
            pytest.skip("Mask R-CNN model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['maskrcnn'],
                model_type='maskrcnn'
            )
            
            assert model is not None
            assert device is not None
            assert transform is not None
            assert not model.training  # Should be in eval mode
            
        except Exception as e:
            pytest.fail(f"Mask R-CNN model present but failed to load: {e}")
    
    def test_load_unet_if_available(self, available_models):
        """Test loading UNet model if available"""
        if 'unet' not in available_models:
            pytest.skip("UNet model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['unet'],
                model_type='unet'
            )
            
            assert model is not None
            assert device is not None
            assert transform is not None
            assert not model.training  # Should be in eval mode
            
        except Exception as e:
            pytest.fail(f"UNet model present but failed to load: {e}")
    
    def test_at_least_one_model_available(self, available_models):
        """Test that at least one model is available"""
        if not available_models:
            pytest.fail("No models found. At least one model (Mask R-CNN or UNet) must be present.")
        
        assert len(available_models) > 0, "No trained models available for testing"


class TestPrediction:
    """Test prediction functionality for available models"""
    
    def test_predict_maskrcnn_if_available(self, available_models, sample_test_image):
        """Test prediction with Mask R-CNN if available"""
        if 'maskrcnn' not in available_models:
            pytest.skip("Mask R-CNN model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['maskrcnn'],
                model_type='maskrcnn'
            )
            
            # Perform prediction
            mask = predict_single_image(
                model, sample_test_image, device, transform,
                model_type='maskrcnn'
            )
            
            # Validate output
            assert mask is not None, "Prediction returned None"
            assert isinstance(mask, np.ndarray), "Prediction should return numpy array"
            assert mask.dtype == np.uint8, "Mask should be uint8"
            assert mask.ndim == 2, "Mask should be 2D"
            assert mask.min() >= 0 and mask.max() <= 255, "Mask values should be in [0, 255]"
            
        except Exception as e:
            pytest.fail(f"Mask R-CNN model present but prediction failed: {e}")
    
    def test_predict_unet_if_available(self, available_models, sample_test_image):
        """Test prediction with UNet if available"""
        if 'unet' not in available_models:
            pytest.skip("UNet model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['unet'],
                model_type='unet'
            )
            
            # Perform prediction
            mask = predict_single_image(
                model, sample_test_image, device, transform,
                model_type='unet'
            )
            
            # Validate output
            assert mask is not None, "Prediction returned None"
            assert isinstance(mask, np.ndarray), "Prediction should return numpy array"
            assert mask.dtype == np.uint8, "Mask should be uint8"
            assert mask.ndim == 2, "Mask should be 2D"
            assert mask.min() >= 0 and mask.max() <= 255, "Mask values should be in [0, 255]"
            
        except Exception as e:
            pytest.fail(f"UNet model present but prediction failed: {e}")
    
    def test_predict_with_timing_maskrcnn(self, available_models, sample_test_image):
        """Test prediction with timing for Mask R-CNN"""
        if 'maskrcnn' not in available_models:
            pytest.skip("Mask R-CNN model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['maskrcnn'],
                model_type='maskrcnn'
            )
            
            # Perform prediction with timing
            prediction_time, mask = predict_single_image(
                model, sample_test_image, device, transform,
                return_time=True, model_type='maskrcnn'
            )
            
            assert prediction_time is not None, "Prediction time should be returned"
            assert prediction_time > 0, "Prediction time should be positive"
            assert mask is not None, "Mask should be returned"
            
        except Exception as e:
            pytest.fail(f"Mask R-CNN model present but timed prediction failed: {e}")
    
    def test_predict_with_timing_unet(self, available_models, sample_test_image):
        """Test prediction with timing for UNet"""
        if 'unet' not in available_models:
            pytest.skip("UNet model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['unet'],
                model_type='unet'
            )
            
            # Perform prediction with timing
            prediction_time, mask = predict_single_image(
                model, sample_test_image, device, transform,
                return_time=True, model_type='unet'
            )
            
            assert prediction_time is not None, "Prediction time should be returned"
            assert prediction_time > 0, "Prediction time should be positive"
            assert mask is not None, "Mask should be returned"
            
        except Exception as e:
            pytest.fail(f"UNet model present but timed prediction failed: {e}")


class TestBothModels:
    """Tests that run when both models are available"""
    
    def test_both_models_produce_output(self, available_models, sample_test_image):
        """Test that both models produce valid output when both are available"""
        if len(available_models) < 2:
            pytest.skip("Both models not available")
        
        results = {}
        
        for model_type, model_path in available_models.items():
            try:
                model, device, transform = load_model(
                    model_path=model_path,
                    model_type=model_type
                )
                
                mask = predict_single_image(
                    model, sample_test_image, device, transform,
                    model_type=model_type
                )
                
                results[model_type] = mask
                
            except Exception as e:
                pytest.fail(f"{model_type} model present but failed: {e}")
        
        # Verify both produced output
        assert 'maskrcnn' in results, "Mask R-CNN should produce output"
        assert 'unet' in results, "UNet should produce output"
        
        # Verify outputs have same shape
        assert results['maskrcnn'].shape == results['unet'].shape, \
            "Both models should produce same shape output"
    
    def test_compare_inference_time(self, available_models, sample_test_image):
        """Compare inference time between models when both available"""
        if len(available_models) < 2:
            pytest.skip("Both models not available")
        
        times = {}
        
        for model_type, model_path in available_models.items():
            try:
                model, device, transform = load_model(
                    model_path=model_path,
                    model_type=model_type
                )
                
                prediction_time, _ = predict_single_image(
                    model, sample_test_image, device, transform,
                    return_time=True, model_type=model_type
                )
                
                times[model_type] = prediction_time
                
            except Exception as e:
                pytest.fail(f"{model_type} model present but timing failed: {e}")
        
        # Just verify both completed successfully
        assert 'maskrcnn' in times, "Mask R-CNN timing should complete"
        assert 'unet' in times, "UNet timing should complete"
        
        print(f"\nInference times:")
        print(f"  Mask R-CNN: {times.get('maskrcnn', 'N/A'):.3f}s")
        print(f"  UNet: {times.get('unet', 'N/A'):.3f}s")


class TestModelConsistency:
    """Test model consistency and output validation"""
    
    def test_maskrcnn_consistent_output(self, available_models, sample_test_image):
        """Test that Mask R-CNN produces consistent output"""
        if 'maskrcnn' not in available_models:
            pytest.skip("Mask R-CNN model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['maskrcnn'],
                model_type='maskrcnn'
            )
            
            # Run prediction twice
            mask1 = predict_single_image(
                model, sample_test_image, device, transform,
                model_type='maskrcnn'
            )
            mask2 = predict_single_image(
                model, sample_test_image, device, transform,
                model_type='maskrcnn'
            )
            
            # Should produce identical results (deterministic)
            np.testing.assert_array_equal(mask1, mask2,
                "Mask R-CNN should produce consistent results")
            
        except Exception as e:
            pytest.fail(f"Mask R-CNN consistency test failed: {e}")
    
    def test_unet_consistent_output(self, available_models, sample_test_image):
        """Test that UNet produces consistent output"""
        if 'unet' not in available_models:
            pytest.skip("UNet model not available")
        
        try:
            model, device, transform = load_model(
                model_path=available_models['unet'],
                model_type='unet'
            )
            
            # Run prediction twice
            mask1 = predict_single_image(
                model, sample_test_image, device, transform,
                model_type='unet'
            )
            mask2 = predict_single_image(
                model, sample_test_image, device, transform,
                model_type='unet'
            )
            
            # Should produce identical results (deterministic)
            np.testing.assert_array_equal(mask1, mask2,
                "UNet should produce consistent results")
            
        except Exception as e:
            pytest.fail(f"UNet consistency test failed: {e}")
