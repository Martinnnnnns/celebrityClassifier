"""
Image Classification - Refactored

This module implements a complete machine learning pipeline for classifying sports celebrities.
Updated to use centralized configuration and corrected color space handling.

Features:
1. Face detection using OpenCV Haar Cascades (flexible detection methods)
2. Wavelet transform feature extraction for enhanced facial features
3. Multiple ML models comparison (SVM, Random Forest, Logistic Regression)
4. Grid search hyperparameter optimization
5. Model persistence and deployment utilities
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import seaborn as sns
import joblib
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, List, Optional, Any

# Add API directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api'))
import config

# Import refactored data cleaning
from data_cleaning_refactored import ImageDataCleaner


class WaveletTransform:
    """Handles wavelet transformation for enhanced facial feature extraction."""

    @staticmethod
    def w2d(img: np.ndarray, mode: str = 'haar', level: int = 1) -> np.ndarray:
        """
        Apply 2D wavelet transform to highlight facial features.

        Args:
            img: Input image
            mode: Wavelet mode ('haar', 'db1', etc.)
            level: Decomposition level

        Returns:
            Wavelet transformed image emphasizing edges
        """
        img_array = img.copy()

        # Convert to grayscale if needed (OpenCV uses BGR, not RGB)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1]
        img_array = np.float32(img_array)
        img_array /= 255.0

        # Apply wavelet decomposition
        coeffs = pywt.wavedec2(img_array, mode, level=level)

        # Zero out approximation coefficients (keep only detail)
        coeffs_h = list(coeffs)
        coeffs_h[0] *= 0

        # Reconstruct from detail coefficients
        img_array_h = pywt.waverec2(coeffs_h, mode)

        # Convert back to [0, 255]
        img_array_h *= 255
        img_array_h = np.uint8(img_array_h)

        return img_array_h

    @staticmethod
    def visualize_wavelet_transform(
        img: np.ndarray,
        mode: str = 'db1',
        level: int = 5
    ) -> None:
        """
        Visualize original vs wavelet transformed image.

        Args:
            img: Input image
            mode: Wavelet mode
            level: Decomposition level
        """
        wavelet_img = WaveletTransform.w2d(img, mode, level)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(wavelet_img, cmap='gray')
        plt.title(f'Wavelet Transform ({mode}, level={level})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


class SportsCelebrityClassifier:
    """Main classifier for sports celebrity image recognition."""

    def __init__(
        self,
        dataset_path: str = "./training_data/",
        cropped_path: str = "./training_data/cropped/"
    ):
        """
        Initialize the classifier with dataset paths.

        Args:
            dataset_path: Path to raw dataset
            cropped_path: Path to processed/cropped dataset
        """
        self.dataset_path = dataset_path
        self.cropped_path = cropped_path
        self.data_cleaner = ImageDataCleaner(dataset_path, cropped_path)
        self.wavelet = WaveletTransform()

        # Use config for celebrities
        self.celebrities = config.CELEBRITIES

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.class_dict: Dict[str, int] = {}
        self.best_model: Optional[Any] = None
        self.celebrity_file_names_dict: Dict[str, List[str]] = {}

        # Load cascades using config paths
        self.face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(config.EYE_CASCADE_PATH)

    def prepare_dataset(self, detection_method: str = 'flexible') -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Prepare the complete dataset: clean images and create class mappings.

        Args:
            detection_method: 'flexible', 'strict', or 'face_only'

        Returns:
            Tuple of (cropped_directories, celebrity_file_names_dict)
        """
        print("="*60)
        print("PREPARING SPORTS CELEBRITY DATASET")
        print("="*60)
        print(f"Celebrities: {', '.join(config.CELEBRITIES)}")
        print(f"Detection method: {detection_method}")
        print()

        cropped_dirs, celebrity_files = self.data_cleaner.process_all_images(
            detection_method=detection_method
        )
        self.celebrity_file_names_dict = celebrity_files

        # Create class dictionary matching config order
        self.class_dict = {name: idx for idx, name in enumerate(config.CELEBRITIES)}

        print(f"\nClass mapping: {self.class_dict}")
        return cropped_dirs, celebrity_files

    def load_existing_dataset(self) -> None:
        """Load dataset from existing cropped folder."""
        cropped_dirs = []
        self.celebrity_file_names_dict = {}

        for entry in os.scandir(self.cropped_path):
            if entry.is_dir():
                celebrity_name = entry.name
                cropped_dirs.append(entry.path)
                file_list = []
                for file_entry in os.scandir(entry.path):
                    if file_entry.is_file():
                        file_list.append(file_entry.path)
                self.celebrity_file_names_dict[celebrity_name] = file_list

        # Create class dictionary matching config order
        self.class_dict = {name: idx for idx, name in enumerate(config.CELEBRITIES)}
        print(f"Loaded dataset with classes: {self.class_dict}")

    def create_feature_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature vectors combining raw pixels and wavelet transform.

        Returns:
            Tuple of (X, y) feature matrix and labels
        """
        print("\nCreating feature vectors...")
        print("Combining raw pixel data with wavelet transform features")
        print(f"Target image size: {config.IMAGE_SIZE}")
        print(f"Wavelet mode: {config.WAVELET_MODE}, level: {config.WAVELET_LEVEL}")

        X, y = [], []
        total_processed = 0

        for celebrity_name, training_files in self.celebrity_file_names_dict.items():
            print(f"Processing {len(training_files)} images for {celebrity_name}")

            for training_image in training_files:
                try:
                    img = cv2.imread(training_image)
                    if img is None:
                        continue

                    # Resize raw image
                    scaled_raw_img = cv2.resize(img, config.IMAGE_SIZE)

                    # Apply wavelet transform (now with correct BGR handling)
                    img_wavelet = self.wavelet.w2d(
                        img,
                        config.WAVELET_MODE,
                        config.WAVELET_LEVEL
                    )
                    scaled_img_wavelet = cv2.resize(img_wavelet, config.IMAGE_SIZE)

                    # Flatten features
                    raw_features = scaled_raw_img.flatten()
                    wavelet_features = scaled_img_wavelet.flatten()

                    # Combine features
                    combined_features = np.concatenate([raw_features, wavelet_features])

                    X.append(combined_features)
                    y.append(self.class_dict[celebrity_name])
                    total_processed += 1

                except Exception as e:
                    print(f"Error processing {training_image}: {e}")
                    continue

        self.X = np.array(X).reshape(len(X), config.TOTAL_FEATURES).astype(float)
        self.y = np.array(y)

        print(f"Created {self.X.shape[0]} feature vectors with {self.X.shape[1]} features each")
        print(f"Feature composition: {config.RAW_PIXEL_FEATURES} raw pixels + {config.WAVELET_FEATURES} wavelet pixels")

        return self.X, self.y

    def train_and_compare_models(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train and compare multiple ML models with hyperparameter tuning.

        Args:
            test_size: Fraction of data for testing (uses config default if None)
            random_state: Random seed (uses config default if None)

        Returns:
            Dictionary of trained models and their performance
        """
        # Use config defaults if not specified
        test_size = test_size or config.TEST_SIZE
        random_state = random_state or config.RANDOM_STATE

        print("\n" + "="*60)
        print("TRAINING AND COMPARING MODELS")
        print("="*60)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Model configurations using config parameters
        model_params = {
            'svm': {
                'model': SVC(gamma='auto', probability=True),
                'params': {
                    'svc__C': config.MODEL_PARAMS['svm']['C'],
                    'svc__kernel': config.MODEL_PARAMS['svm']['kernel']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'randomforestclassifier__n_estimators': config.MODEL_PARAMS['random_forest']['n_estimators']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(solver='liblinear', multi_class='auto'),
                'params': {
                    'logisticregression__C': config.MODEL_PARAMS['logistic_regression']['C']
                }
            }
        }

        scores = []
        best_estimators = {}

        for algo, mp in model_params.items():
            print(f"\nTraining {algo}...")

            pipe = make_pipeline(StandardScaler(), mp['model'])

            clf = GridSearchCV(
                pipe,
                mp['params'],
                cv=config.CV_FOLDS,
                return_train_score=False
            )
            clf.fit(X_train, y_train)

            scores.append({
                'model': algo,
                'best_score': clf.best_score_,
                'best_params': clf.best_params_,
                'test_score': clf.best_estimator_.score(X_test, y_test)
            })
            best_estimators[algo] = clf.best_estimator_

            print(f"Best CV score: {clf.best_score_:.4f}")
            print(f"Test score: {clf.best_estimator_.score(X_test, y_test):.4f}")
            print(f"Best params: {clf.best_params_}")

        results_df = pd.DataFrame(scores)
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(results_df.to_string(index=False))

        best_model_name = results_df.loc[results_df['test_score'].idxmax(), 'model']
        self.best_model = best_estimators[best_model_name]

        print(f"\nBest performing model: {best_model_name}")
        print(f"Test accuracy: {results_df['test_score'].max():.4f}")

        self._evaluate_model(X_test, y_test)

        return {
            'results': results_df,
            'best_estimators': best_estimators,
            'best_model': self.best_model,
            'test_data': (X_test, y_test)
        }

    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Detailed evaluation of the best model.

        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "="*60)
        print("DETAILED MODEL EVALUATION")
        print("="*60)

        y_pred = self.best_model.predict(X_test)

        target_names = list(self.class_dict.keys())
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.title('Confusion Matrix - Sports Celebrity Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def save_model(
        self,
        model_path: Optional[str] = None,
        class_dict_path: Optional[str] = None
    ) -> None:
        """
        Save the trained model and class dictionary.

        Args:
            model_path: Path to save the model (uses config default if None)
            class_dict_path: Path to save class dictionary (uses config default if None)
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Run train_and_compare_models() first.")

        # Use config defaults if not specified
        model_path = model_path or config.MODEL_PATH
        class_dict_path = class_dict_path or config.CLASS_DICT_PATH

        # Ensure directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(class_dict_path), exist_ok=True)

        joblib.dump(self.best_model, model_path)
        print(f"Model saved to: {model_path}")

        with open(class_dict_path, 'w') as f:
            json.dump(self.class_dict, f, indent=2)
        print(f"Class dictionary saved to: {class_dict_path}")

    def load_model(
        self,
        model_path: Optional[str] = None,
        class_dict_path: Optional[str] = None
    ) -> None:
        """
        Load a previously trained model.

        Args:
            model_path: Path to the saved model (uses config default if None)
            class_dict_path: Path to the class dictionary (uses config default if None)
        """
        # Use config defaults if not specified
        model_path = model_path or config.MODEL_PATH
        class_dict_path = class_dict_path or config.CLASS_DICT_PATH

        self.best_model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")

        with open(class_dict_path, 'r') as f:
            self.class_dict = json.load(f)
        print(f"Class dictionary loaded from: {class_dict_path}")

    def predict_celebrity(
        self,
        image_path: str,
        return_probabilities: bool = False
    ) -> Any:
        """
        Predict celebrity from a new image.

        Args:
            image_path: Path to the image
            return_probabilities: Whether to return prediction probabilities

        Returns:
            Celebrity name or (name, probabilities)
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Train a model or load an existing one.")

        cropped_img = self.data_cleaner.get_cropped_image_with_eye_validation(image_path)
        if cropped_img is None:
            return "No face detected"

        # Extract features using config parameters
        scaled_raw_img = cv2.resize(cropped_img, config.IMAGE_SIZE)
        img_wavelet = self.wavelet.w2d(
            cropped_img,
            config.WAVELET_MODE,
            config.WAVELET_LEVEL
        )
        scaled_img_wavelet = cv2.resize(img_wavelet, config.IMAGE_SIZE)

        # Flatten and combine features
        raw_features = scaled_raw_img.flatten()
        wavelet_features = scaled_img_wavelet.flatten()
        combined_features = np.concatenate([raw_features, wavelet_features])

        feature_vector = combined_features.reshape(1, -1).astype(float)

        prediction = self.best_model.predict(feature_vector)[0]
        celebrity_name = list(self.class_dict.keys())[
            list(self.class_dict.values()).index(prediction)
        ]

        if return_probabilities:
            probabilities = self.best_model.predict_proba(feature_vector)[0]
            prob_dict = {
                name: prob for name, prob in zip(self.class_dict.keys(), probabilities)
            }
            return celebrity_name, prob_dict

        return celebrity_name

    def demonstrate_preprocessing(self, sample_image_path: str) -> None:
        """
        Demonstrate the preprocessing pipeline on a sample image.

        Args:
            sample_image_path: Path to a sample image
        """
        print("Demonstrating preprocessing pipeline...")
        original_img = cv2.imread(sample_image_path)
        if original_img is None:
            print(f"Could not load image: {sample_image_path}")
            return

        cropped_img = self.data_cleaner.get_cropped_image_with_eye_validation(
            sample_image_path
        )

        if cropped_img is None:
            print("No valid face detected in the image")
            return

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Face (Flexible Detection)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        wavelet_img = self.wavelet.w2d(
            cropped_img,
            config.WAVELET_MODE,
            config.WAVELET_LEVEL
        )
        plt.imshow(wavelet_img, cmap='gray')
        plt.title('Wavelet Transform (Edge Features)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the complete sports celebrity classification pipeline."""
    print("Sports Celebrity Image Classification - Refactored")
    print(f"Celebrities: {', '.join(config.CELEBRITIES)}")
    print(f"Using configuration from: {config.__file__}")
    print()

    classifier = SportsCelebrityClassifier()

    try:
        classifier.load_existing_dataset()
        print("Loaded existing processed dataset")
    except Exception as e:
        print(f"Could not load existing dataset: {e}")
        print("Processing raw dataset...")
        classifier.prepare_dataset()

    X, y = classifier.create_feature_vectors()
    results = classifier.train_and_compare_models()
    classifier.save_model()

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Model and class dictionary saved.")
    print("You can now use the classifier to predict celebrities in new images.")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Class dictionary saved to: {config.CLASS_DICT_PATH}")

    return classifier, results


if __name__ == "__main__":
    classifier, training_results = main()
