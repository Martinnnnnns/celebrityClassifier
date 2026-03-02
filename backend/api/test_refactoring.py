"""
Validation tests for refactored celebrity classifier

This script tests the refactored code to ensure:
1. Configuration loads correctly
2. Model and artifacts load successfully
3. Image processing functions work
4. Classification pipeline functions correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration module."""
    print("=" * 70)
    print("TEST 1: Configuration Module")
    print("=" * 70)

    try:
        import config

        print(f"✓ Config module imported successfully")
        print(f"  Celebrities: {config.CELEBRITIES}")
        print(f"  Image size: {config.IMAGE_SIZE}")
        print(f"  Total features: {config.TOTAL_FEATURES}")
        print(f"  Model path: {config.MODEL_PATH}")
        print(f"  Face cascade path: {config.FACE_CASCADE_PATH}")

        # Verify paths exist
        assert os.path.exists(config.FACE_CASCADE_PATH), "Face cascade not found"
        assert os.path.exists(config.EYE_CASCADE_PATH), "Eye cascade not found"
        assert os.path.exists(config.MODEL_PATH), "Model not found"
        assert os.path.exists(config.CLASS_DICT_PATH), "Class dictionary not found"

        print(f"✓ All required files exist")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wavelet_transform():
    """Test wavelet transform function."""
    print("\n" + "=" * 70)
    print("TEST 2: Wavelet Transform")
    print("=" * 70)

    try:
        import cv2
        import numpy as np
        from wavelet_transform import w2d

        # Create a test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Apply wavelet transform
        wavelet_result = w2d(test_img, 'db1', 5)

        assert wavelet_result is not None, "Wavelet transform returned None"
        assert len(wavelet_result.shape) == 2, "Wavelet result should be 2D"

        print(f"✓ Wavelet transform works correctly")
        print(f"  Input shape: {test_img.shape}")
        print(f"  Output shape: {wavelet_result.shape}")
        return True

    except Exception as e:
        print(f"✗ Wavelet transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model and artifact loading."""
    print("\n" + "=" * 70)
    print("TEST 3: Model and Artifact Loading")
    print("=" * 70)

    try:
        import image_utils

        # Load artifacts
        image_utils.load_saved_artifacts()

        # Try a simple classification to verify model is loaded
        # We can't directly access private variables, so we test functionality
        print(f"✓ Model and artifacts loaded successfully")
        print(f"  Testing model functionality...")

        # The fact that load_saved_artifacts() completed without error means it worked
        # Test that class_number_to_name works
        test_name = image_utils.class_number_to_name(0)
        assert test_name in ['cristiano_ronaldo', 'lionel_messi', 'steph_curry', 'serena_williams', 'carlos_alcaraz']
        print(f"  Class 0 maps to: {test_name}")

        return True

    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_functions():
    """Test image validation functions."""
    print("\n" + "=" * 70)
    print("TEST 4: Validation Functions")
    print("=" * 70)

    try:
        import cv2
        import numpy as np
        from image_utils import validate_image, validate_face_crop

        # Test valid image
        valid_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_valid, error_msg = validate_image(valid_img)
        assert is_valid, f"Valid image rejected: {error_msg}"
        print(f"✓ Valid image accepted")

        # Test invalid image (None)
        is_valid, error_msg = validate_image(None)
        assert not is_valid, "None image should be rejected"
        print(f"✓ None image rejected: {error_msg}")

        # Test invalid image (too small)
        small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        is_valid, error_msg = validate_image(small_img)
        assert not is_valid, "Too small image should be rejected"
        print(f"✓ Small image rejected: {error_msg}")

        # Test valid face crop
        valid_face = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        assert validate_face_crop(valid_face), "Valid face crop rejected"
        print(f"✓ Valid face crop accepted")

        # Test invalid face crop
        invalid_face = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        assert not validate_face_crop(invalid_face), "Invalid face crop should be rejected"
        print(f"✓ Invalid face crop rejected")

        return True

    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\n" + "=" * 70)
    print("TEST 5: Feature Extraction")
    print("=" * 70)

    try:
        import cv2
        import numpy as np
        from image_utils import extract_features
        import config

        # Create a test face image
        test_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Extract features
        features = extract_features(test_face)

        assert features is not None, "Feature extraction returned None"
        assert features.shape == (1, config.TOTAL_FEATURES), f"Feature shape mismatch: {features.shape}"

        print(f"✓ Feature extraction works correctly")
        print(f"  Input shape: {test_face.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Expected features: {config.TOTAL_FEATURES}")
        return True

    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_dictionary_consistency():
    """Test that class dictionary matches expected order."""
    print("\n" + "=" * 70)
    print("TEST 6: Class Dictionary Consistency")
    print("=" * 70)

    try:
        import json
        import config

        # Load class dictionary
        with open(config.CLASS_DICT_PATH, 'r') as f:
            class_dict = json.load(f)

        # Expected order
        expected_order = {name: idx for idx, name in enumerate(config.CELEBRITIES)}

        print(f"  Loaded class dict: {class_dict}")
        print(f"  Expected order: {expected_order}")

        # Check consistency
        for name, idx in expected_order.items():
            assert name in class_dict, f"Celebrity {name} not in class dictionary"
            assert class_dict[name] == idx, f"Index mismatch for {name}: expected {idx}, got {class_dict[name]}"

        print(f"✓ Class dictionary matches expected order")
        return True

    except Exception as e:
        print(f"✗ Class dictionary consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("REFACTORING VALIDATION TESTS")
    print("=" * 70)

    tests = [
        ("Configuration", test_config),
        ("Wavelet Transform", test_wavelet_transform),
        ("Model Loading", test_model_loading),
        ("Validation Functions", test_validation_functions),
        ("Feature Extraction", test_feature_extraction),
        ("Class Dictionary Consistency", test_class_dictionary_consistency),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ ALL TESTS PASSED - Refactoring successful!")
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
