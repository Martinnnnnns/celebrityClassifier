"""
Refactored Image Utilities for Celebrity Classification

This module provides improved image processing functionality with:
- Centralized configuration
- Input validation
- Type hints
- Better error handling
- Cleaner code structure
"""

import joblib
import json
import numpy as np
import base64
import cv2
from typing import List, Dict, Optional, Tuple, Any
from wavelet_transform import w2d
import config
import os

# ============================================================================
# MODULE-LEVEL STATE
# ============================================================================

__class_name_to_number: Dict[str, int] = {}
__class_number_to_name: Dict[int, str] = {}
__model: Optional[Any] = None


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_image(img: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate image for processing.

    Args:
        img: Input image array

    Returns:
        Tuple of (is_valid, error_message)
    """
    if img is None:
        return False, "Image is None"

    if not isinstance(img, np.ndarray):
        return False, "Image is not a numpy array"

    if img.size == 0:
        return False, "Image is empty"

    # Check dimensions
    if len(img.shape) < 2:
        return False, "Image has invalid dimensions"

    height, width = img.shape[:2]

    # Check if too small
    if height < config.MIN_FACE_SIZE[1] or width < config.MIN_FACE_SIZE[0]:
        return False, f"Image too small (min: {config.MIN_FACE_SIZE})"

    # Check if too large
    if height > config.MAX_IMAGE_HEIGHT or width > config.MAX_IMAGE_WIDTH:
        return False, f"Image too large (max: {config.MAX_IMAGE_WIDTH}x{config.MAX_IMAGE_HEIGHT})"

    # Check color channels
    if len(img.shape) == 3 and img.shape[2] not in [3, 4]:
        return False, f"Unsupported number of color channels: {img.shape[2]}"

    return True, None


def validate_face_crop(face: np.ndarray) -> bool:
    """
    Validate a cropped face region.

    Args:
        face: Cropped face image

    Returns:
        True if valid, False otherwise
    """
    if face is None or face.size == 0:
        return False

    height, width = face.shape[:2]

    return (height >= config.MIN_FACE_SIZE[1] and
            width >= config.MIN_FACE_SIZE[0])


# ============================================================================
# IMAGE DECODING
# ============================================================================

def decode_base64_image(b64str: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to OpenCV image with validation.

    Args:
        b64str: Base64 encoded image string (with or without data URI prefix)

    Returns:
        OpenCV image array or None if decoding fails
    """
    try:
        # Remove data URI prefix if present
        if ',' in b64str:
            encoded_data = b64str.split(',', 1)[1]
        else:
            encoded_data = b64str

        # Decode base64 to bytes
        img_bytes = base64.b64decode(encoded_data)

        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Validate decoded image
        is_valid, error_msg = validate_image(img)
        if not is_valid:
            print(f"Image validation failed: {error_msg}")
            return None

        return img

    except base64.binascii.Error as e:
        print(f"Base64 decoding error: {e}")
        return None
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


# ============================================================================
# FACE DETECTION
# ============================================================================

def detect_faces_with_strategy(
    gray: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
    img: np.ndarray
) -> List[np.ndarray]:
    """
    Detect faces using multiple fallback strategies.

    Strategy 1: Standard detection with 2+ eyes (highest quality)
    Strategy 2: Standard detection with 1+ eye (medium quality)
    Strategy 3: Lenient detection with 1+ eye (lower quality)
    Strategy 4: Very lenient detection, largest face (lowest quality)

    Args:
        gray: Grayscale image
        face_cascade: Face detection cascade
        eye_cascade: Eye detection cascade
        img: Original color image

    Returns:
        List of cropped face images
    """
    cropped_faces = []

    if config.VERBOSE_LOGGING:
        print("Starting flexible face detection...")

    # Strategy 1: Standard detection with 2+ eyes (highest quality)
    faces = face_cascade.detectMultiScale(
        gray,
        **config.DETECTION_PARAMS_STANDARD
    )

    if config.VERBOSE_LOGGING:
        print(f"Strategy 1 - Standard detection: Found {len(faces)} faces")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if config.VERBOSE_LOGGING:
            print(f"  Face at ({x},{y},{w},{h}) has {len(eyes)} eyes")

        if len(eyes) >= config.MIN_EYES_HIGH_QUALITY:
            cropped_face = extract_face_with_padding(img, x, y, w, h)
            if validate_face_crop(cropped_face):
                cropped_faces.append(cropped_face)

    if cropped_faces:
        if config.VERBOSE_LOGGING:
            print(f"SUCCESS: Found {len(cropped_faces)} high-quality faces (2+ eyes)")
        return cropped_faces

    # Strategy 2: Standard detection with 1+ eye (medium quality)
    if config.VERBOSE_LOGGING:
        print("Strategy 2 - Relaxed eye requirement (1+ eye)")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= config.MIN_EYES_MEDIUM_QUALITY:
            cropped_face = extract_face_with_padding(img, x, y, w, h)
            if validate_face_crop(cropped_face):
                cropped_faces.append(cropped_face)

    if cropped_faces:
        if config.VERBOSE_LOGGING:
            print(f"SUCCESS: Found {len(cropped_faces)} medium-quality faces (1+ eye)")
        return cropped_faces

    # Strategy 3: Lenient detection with 1+ eye
    if config.VERBOSE_LOGGING:
        print("Strategy 3 - Lenient face detection")

    faces = face_cascade.detectMultiScale(
        gray,
        **config.DETECTION_PARAMS_LENIENT
    )

    if config.VERBOSE_LOGGING:
        print(f"  Lenient detection: Found {len(faces)} faces")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= config.MIN_EYES_MEDIUM_QUALITY:
            cropped_face = extract_face_with_padding(img, x, y, w, h)
            if validate_face_crop(cropped_face):
                cropped_faces.append(cropped_face)

    if cropped_faces:
        if config.VERBOSE_LOGGING:
            print(f"SUCCESS: Found {len(cropped_faces)} faces with lenient detection")
        return cropped_faces

    # Strategy 4: Very lenient detection, largest face
    if config.VERBOSE_LOGGING:
        print("Strategy 4 - Very lenient face detection")

    faces = face_cascade.detectMultiScale(
        gray,
        **config.DETECTION_PARAMS_VERY_LENIENT
    )

    if config.VERBOSE_LOGGING:
        print(f"  Very lenient detection: Found {len(faces)} faces")

    if len(faces) > 0:
        # Take the largest face (most likely to be the main subject)
        areas = [w * h for (x, y, w, h) in faces]
        largest_face_idx = np.argmax(areas)
        x, y, w, h = faces[largest_face_idx]

        cropped_face = extract_face_with_padding(img, x, y, w, h)

        if validate_face_crop(cropped_face):
            if config.VERBOSE_LOGGING:
                print(f"SUCCESS: Using largest face (no eye requirement)")
            return [cropped_face]

    if config.VERBOSE_LOGGING:
        print("FAILED: No faces detected with any strategy")

    return []


def extract_face_with_padding(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int
) -> np.ndarray:
    """
    Extract face region with padding.

    Args:
        img: Source image
        x, y, w, h: Face bounding box coordinates

    Returns:
        Cropped face image with padding
    """
    padding = int(config.FACE_PADDING_RATIO * min(w, h))

    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)

    return img[y_start:y_end, x_start:x_end]


def get_cropped_faces(
    image_path: Optional[str] = None,
    image_base64_data: Optional[str] = None
) -> List[np.ndarray]:
    """
    Detect and crop faces from an image using flexible detection strategies.

    Args:
        image_path: Path to image file (optional)
        image_base64_data: Base64 encoded image data (optional)

    Returns:
        List of cropped face images
    """
    # Load cascades
    face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(config.EYE_CASCADE_PATH)

    # Validate cascades loaded successfully
    if face_cascade.empty() or eye_cascade.empty():
        print("ERROR: Failed to load cascade classifiers")
        return []

    # Load image
    if image_path:
        img = cv2.imread(image_path)
    elif image_base64_data:
        img = decode_base64_image(image_base64_data)
    else:
        print("ERROR: No image source provided")
        return []

    # Validate image
    is_valid, error_msg = validate_image(img)
    if not is_valid:
        print(f"ERROR: {error_msg}")
        return []

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces with multiple strategies
    return detect_faces_with_strategy(gray, face_cascade, eye_cascade, img)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(face_img: np.ndarray) -> np.ndarray:
    """
    Extract combined features from a face image.

    Combines raw pixel features with wavelet transform features for
    enhanced accuracy.

    Args:
        face_img: Cropped face image

    Returns:
        Feature vector of shape (1, TOTAL_FEATURES)
    """
    # Resize to standard size
    resized_raw = cv2.resize(face_img, config.IMAGE_SIZE)

    # Apply wavelet transform
    wavelet_img = w2d(face_img, config.WAVELET_MODE, config.WAVELET_LEVEL)
    resized_wavelet = cv2.resize(wavelet_img, config.IMAGE_SIZE)

    # Flatten features
    raw_features = resized_raw.flatten()
    wavelet_features = resized_wavelet.flatten()

    # Concatenate features
    combined_features = np.concatenate([raw_features, wavelet_features])

    # Reshape to (1, n_features) for prediction
    return combined_features.reshape(1, -1).astype(float)


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_image(
    image_base64_data: Optional[str] = None,
    file_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Classify celebrity from image using flexible face detection.

    Args:
        image_base64_data: Base64 encoded image data
        file_path: Optional file path for image

    Returns:
        List of classification results with probabilities
    """
    if __model is None:
        raise RuntimeError("Model not loaded. Call load_saved_artifacts() first.")

    # Detect and crop faces
    cropped_faces = get_cropped_faces(file_path, image_base64_data)

    if not cropped_faces:
        return []

    results = []

    for idx, face_img in enumerate(cropped_faces):
        try:
            # Extract features
            features = extract_features(face_img)

            # Validate feature dimensions
            if features.shape[1] != config.TOTAL_FEATURES:
                print(f"ERROR: Feature dimension mismatch. Expected {config.TOTAL_FEATURES}, got {features.shape[1]}")
                continue

            # Make prediction
            prediction = __model.predict(features)[0]
            probabilities = __model.predict_proba(features)[0]

            # Get celebrity name
            celebrity_name = class_number_to_name(prediction)

            # Convert probabilities to percentages
            probabilities_pct = np.around(probabilities * 100, 2).tolist()

            results.append({
                'class': celebrity_name,
                'class_probability': probabilities_pct,
                'class_dictionary': __class_name_to_number,
                'detection_info': {
                    'method': 'flexible_detection',
                    'face_number': idx + 1,
                    'total_faces': len(cropped_faces)
                }
            })

        except Exception as e:
            print(f"Error processing face {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def class_number_to_name(class_num: int) -> str:
    """
    Convert class number to celebrity name.

    Args:
        class_num: Class index

    Returns:
        Celebrity name
    """
    if class_num not in __class_number_to_name:
        raise ValueError(f"Unknown class number: {class_num}")

    return __class_number_to_name[class_num]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_saved_artifacts() -> None:
    """
    Load saved model and class dictionary from disk.

    Raises:
        FileNotFoundError: If model or class dictionary not found
        RuntimeError: If loading fails
    """
    global __class_name_to_number, __class_number_to_name, __model

    print("Loading saved artifacts...")

    # Validate paths exist
    if not os.path.exists(config.CLASS_DICT_PATH):
        raise FileNotFoundError(f"Class dictionary not found at: {config.CLASS_DICT_PATH}")

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {config.MODEL_PATH}")

    # Load class dictionary
    try:
        with open(config.CLASS_DICT_PATH, "r") as f:
            __class_name_to_number = json.load(f)
            __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
        print(f"✓ Class dictionary loaded from: {config.CLASS_DICT_PATH}")
        print(f"  Classes: {list(__class_name_to_number.keys())}")
    except Exception as e:
        raise RuntimeError(f"Failed to load class dictionary: {e}")

    # Load model
    try:
        with open(config.MODEL_PATH, 'rb') as f:
            __model = joblib.load(f)
        print(f"✓ Model loaded from: {config.MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    print("Artifacts loaded successfully")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    load_saved_artifacts()
    print("\nTesting classification (uncomment test images below):")
    # print(classify_image(None, "./test_images/ronaldo.jpeg"))
    # print(classify_image(None, "./test_images/messi.jpeg"))
