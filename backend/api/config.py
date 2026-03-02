"""
Configuration module for Celebrity Classifier

Centralizes all configuration parameters, paths, and constants for easier
maintenance and tuning. This eliminates magic numbers throughout the codebase.
"""

import os
from typing import Tuple

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Resource paths
RESOURCES_DIR = os.path.join(BACKEND_DIR, 'resources')
OPENCV_DIR = os.path.join(RESOURCES_DIR, 'opencv', 'haarcascades')

# Model paths
MODELS_DIR = os.path.join(BACKEND_DIR, 'models')
ARTIFACTS_DIR = os.path.join(MODELS_DIR, 'saved_artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'saved_model.pkl')
CLASS_DICT_PATH = os.path.join(ARTIFACTS_DIR, 'class_dictionary.json')

# Cascade file paths
FACE_CASCADE_PATH = os.path.join(OPENCV_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(OPENCV_DIR, 'haarcascade_eye.xml')

# ============================================================================
# CELEBRITY CONFIGURATION
# ============================================================================

# Celebrity names in consistent order (used for training and prediction)
CELEBRITIES = [
    'cristiano_ronaldo',
    'lionel_messi',
    'steph_curry',
    'serena_williams',
    'carlos_alcaraz'
]

# Display names for frontend
CELEBRITY_DISPLAY_NAMES = {
    'cristiano_ronaldo': 'Cristiano Ronaldo',
    'lionel_messi': 'Lionel Messi',
    'steph_curry': 'Steph Curry',
    'serena_williams': 'Serena Williams',
    'carlos_alcaraz': 'Carlos Alcaraz'
}

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================

# Target image size for feature extraction (width, height)
IMAGE_SIZE: Tuple[int, int] = (32, 32)

# Face detection padding (as fraction of face dimension)
FACE_PADDING_RATIO: float = 0.1

# Minimum face dimensions for quality check (width, height in pixels)
MIN_FACE_SIZE: Tuple[int, int] = (30, 30)

# Maximum image dimensions for processing (to prevent memory issues)
MAX_IMAGE_WIDTH: int = 4096
MAX_IMAGE_HEIGHT: int = 4096

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

# ============================================================================
# FACE DETECTION PARAMETERS
# ============================================================================

# Face detection cascade parameters (scaleFactor, minNeighbors)
# Strategy 1: Standard detection (highest quality)
DETECTION_PARAMS_STANDARD = {
    'scaleFactor': 1.3,
    'minNeighbors': 5
}

# Strategy 2: Lenient detection (medium quality)
DETECTION_PARAMS_LENIENT = {
    'scaleFactor': 1.1,
    'minNeighbors': 3
}

# Strategy 3: Very lenient detection (lowest quality, highest coverage)
DETECTION_PARAMS_VERY_LENIENT = {
    'scaleFactor': 1.05,
    'minNeighbors': 2
}

# Minimum number of eyes required for high-quality face detection
MIN_EYES_HIGH_QUALITY: int = 2
MIN_EYES_MEDIUM_QUALITY: int = 1

# ============================================================================
# WAVELET TRANSFORM PARAMETERS
# ============================================================================

# Wavelet mode for feature extraction
WAVELET_MODE: str = 'db1'  # Daubechies wavelet

# Decomposition level for wavelet transform
WAVELET_LEVEL: int = 5

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# Calculate feature dimensions based on image size
RAW_PIXEL_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3  # 32*32*3 = 3072
WAVELET_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1]        # 32*32 = 1024
TOTAL_FEATURES = RAW_PIXEL_FEATURES + WAVELET_FEATURES  # 4096

# ============================================================================
# API CONFIGURATION
# ============================================================================

# CORS allowed origins
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# Maximum content length (32 MB)
MAX_CONTENT_LENGTH = 32 * 1024 * 1024

# Maximum form memory size (16 MB)
MAX_FORM_MEMORY = 16 * 1024 * 1024

# API host and port
API_HOST = '127.0.0.1'
API_PORT = 5000

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Train/test split ratio
TEST_SIZE = 0.25

# Random seed for reproducibility
RANDOM_STATE = 42

# Cross-validation folds
CV_FOLDS = 5

# Model hyperparameter grids
MODEL_PARAMS = {
    'svm': {
        'C': [1, 10, 100, 1000],
        'kernel': ['rbf', 'linear']
    },
    'random_forest': {
        'n_estimators': [1, 5, 10, 20]
    },
    'logistic_regression': {
        'C': [1, 5, 10]
    }
}

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

# Minimum confidence threshold for predictions (0-100%)
MIN_CONFIDENCE_THRESHOLD = 30.0

# Maximum file size for upload (5 MB in bytes)
MAX_UPLOAD_SIZE = 5 * 1024 * 1024

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Enable/disable verbose logging
VERBOSE_LOGGING = True

# Enable/disable debug mode
DEBUG_MODE = False
