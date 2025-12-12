# Prediction Tests

This directory contains tests for the prediction functionality that support both Mask R-CNN and UNet models.

## Test Logic

The prediction tests follow these rules:

1. **If both models are present**: Test both models
   - If either model fails, the test fails
   - Both models must produce valid output

2. **If only one model is present**: Test that model only
   - If the present model fails, the test fails
   - The absent model's tests are skipped

3. **If no models are present**: Fail the test
   - At least one model must be available

## Running Tests

### Run all prediction tests:
```bash
pytest tests/test_prediction.py -v
```

### Run only tests that work with available models:
```bash
pytest tests/test_prediction.py -v --tb=short
```

### Check which models are available:
```bash
python verify_test_logic.py
```

## Test Classes

### `TestModelInitialization`
Tests that both model architectures can be initialized correctly.

### `TestModelLoading`
Tests loading of trained models from disk:
- Skips if model not available
- Fails if model present but fails to load

### `TestPrediction`
Tests prediction functionality:
- Skips if model not available
- Fails if model present but prediction fails
- Validates output format and values

### `TestBothModels`
Tests that only run when both models are available:
- Compares outputs between models
- Compares inference times
- Skips if both models not available

### `TestModelConsistency`
Tests that models produce consistent (deterministic) output:
- Runs same prediction twice
- Verifies identical results

## Expected Behavior

### Scenario 1: Both models available
```
✅ test_initialize_maskrcnn - PASSED
✅ test_initialize_unet - PASSED
✅ test_load_maskrcnn_if_available - PASSED
✅ test_load_unet_if_available - PASSED
✅ test_predict_maskrcnn_if_available - PASSED
✅ test_predict_unet_if_available - PASSED
✅ test_both_models_produce_output - PASSED
✅ test_compare_inference_time - PASSED
```

### Scenario 2: Only Mask R-CNN available
```
✅ test_initialize_maskrcnn - PASSED
✅ test_initialize_unet - PASSED
✅ test_load_maskrcnn_if_available - PASSED
⚠️  test_load_unet_if_available - SKIPPED (model not available)
✅ test_predict_maskrcnn_if_available - PASSED
⚠️  test_predict_unet_if_available - SKIPPED (model not available)
⚠️  test_both_models_produce_output - SKIPPED (both not available)
⚠️  test_compare_inference_time - SKIPPED (both not available)
```

### Scenario 3: Model present but fails
```
✅ test_initialize_maskrcnn - PASSED
❌ test_load_maskrcnn_if_available - FAILED (model present but failed to load)
```

### Scenario 4: No models available
```
❌ test_at_least_one_model_available - FAILED (no models found)
```

## CI/CD Integration

The GitHub Actions workflow (`test-pipeline.yml`) has been updated to:

1. Check for available models before running predictions
2. Run prediction with each available model
3. Fail if a present model fails
4. Skip if a model is not available
5. Fail if no models are found

## Adding New Tests

When adding new prediction tests:

1. Use the `available_models` fixture to check model availability
2. Skip tests if required model not available: `pytest.skip("Model not available")`
3. Fail tests if model present but fails: `pytest.fail("Model present but failed")`
4. Always validate output format and values
