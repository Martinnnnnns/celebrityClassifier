"""
Image Classification

This module implements a complete machine learning pipeline for classifying sports celebrities:
- Cristiano Ronaldo (Football)
- Lionel Messi (Football)
- Steph Curry (Basketball) 
- Serena Williams (Tennis)
- Carlos Alcaraz(Tennis)

Features:
1. Face detection using OpenCV Haar Cascades (flexible detection methods)
2. Wavelet transform feature extraction for enhanced facial features
3. Multiple ML models comparison (SVM, Random Forest, Logistic Regression)
4. Grid search hyperparameter optimization
5. Model persistence and deployment utilities
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import os
import pandas as pd
import seaborn as sns
import joblib
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from data_cleaning import ImageDataCleaner


class WaveletTransform:
    """Handles wavelet transformation for enhanced facial feature extraction."""
    
    @staticmethod
    def w2d(img, mode='haar', level=1):
        """
        Apply 2D wavelet transform to highlight facial features.
        
        Args:
            img (numpy.ndarray): Input image
            mode (str): Wavelet mode ('haar', 'db1', etc.)
            level (int): Decomposition level
            
        Returns:
            numpy.ndarray: Wavelet transformed image emphasizing edges
        """
        img_array = img.copy()
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Convert to float and normalize
        img_array = np.float32(img_array)
        img_array /= 255.0
        
        # Compute wavelet coefficients
        coeffs = pywt.wavedec2(img_array, mode, level=level)
        
        # Process coefficients - zero out approximation to emphasize details
        coeffs_h = list(coeffs)
        coeffs_h[0] *= 0
        
        # Reconstruct image
        img_array_h = pywt.waverec2(coeffs_h, mode)
        img_array_h *= 255
        img_array_h = np.uint8(img_array_h)
        
        return img_array_h
    
    @staticmethod
    def visualize_wavelet_transform(img, mode='db1', level=5):
        """
        Visualize original vs wavelet transformed image.
        
        Args:
            img (numpy.ndarray): Input image
            mode (str): Wavelet mode
            level (int): Decomposition level
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
    
    def __init__(self, dataset_path="./training_data/", cropped_path="./training_data/cropped/"):
        """
        Initialize the classifier with dataset paths.
        
        Args:
            dataset_path (str): Path to raw dataset
            cropped_path (str): Path to processed/cropped dataset
        """
        self.dataset_path = dataset_path
        self.cropped_path = cropped_path
        self.data_cleaner = ImageDataCleaner(dataset_path, cropped_path)
        self.wavelet = WaveletTransform()
        
        # Celebrity mapping
        self.celebrities = [
            'cristiano_ronaldo', 'lionel_messi', 'steph_curry', 
            'serena_williams', 'carlos_alcaraz'
        ]
        
        # Initialize variables
        self.X = None
        self.y = None
        self.class_dict = {}
        self.best_model = None
        self.celebrity_file_names_dict = {}
        
        # OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier('../../resources/opencv/haarcascades/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('../../resources/opencv/haarcascades/haarcascade_eye.xml')
    
    def prepare_dataset(self, detection_method='flexible'):
        """
        Prepare the complete dataset: clean images and create class mappings.
        
        Args:
            detection_method (str): 'flexible', 'strict', or 'face_only'
        
        Returns:
            tuple: (cropped_directories, celebrity_file_names_dict)
        """
        print("="*60)
        print("PREPARING SPORTS CELEBRITY DATASET")
        print("="*60)
        print("Celebrities: Cristiano Ronaldo, Lionel Messi, Steph Curry,")
        print("             Serena Williams, Carlos Alcaraz")
        print(f"Detection method: {detection_method}")
        print()
        
        # Process and crop images
        cropped_dirs, celebrity_files = self.data_cleaner.process_all_images(detection_method=detection_method)
        self.celebrity_file_names_dict = celebrity_files
        
        # Create class dictionary
        self.class_dict = {name: idx for idx, name in enumerate(self.celebrity_file_names_dict.keys())}
        
        print(f"\nClass mapping: {self.class_dict}")
        return cropped_dirs, celebrity_files
    
    def load_existing_dataset(self):
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
        
        # Create class dictionary
        self.class_dict = {name: idx for idx, name in enumerate(self.celebrity_file_names_dict.keys())}
        print(f"Loaded dataset with classes: {self.class_dict}")
    
    def create_feature_vectors(self, image_size=(32, 32)):
        """
        Create feature vectors combining raw pixels and wavelet transform.
        
        Args:
            image_size (tuple): Target size for resizing images
            
        Returns:
            tuple: (X, y) feature matrix and labels
        """
        print("\nCreating feature vectors...")
        print("Combining raw pixel data with wavelet transform features")
        
        X, y = [], []
        total_processed = 0
        
        for celebrity_name, training_files in self.celebrity_file_names_dict.items():
            print(f"Processing {len(training_files)} images for {celebrity_name}")
            
            for training_image in training_files:
                try:
                    # Load and resize image
                    img = cv2.imread(training_image)
                    if img is None:
                        continue
                    
                    # Resize raw image
                    scaled_raw_img = cv2.resize(img, image_size)
                    
                    # Apply wavelet transform and resize
                    img_wavelet = self.wavelet.w2d(img, 'db1', 5)
                    scaled_img_wavelet = cv2.resize(img_wavelet, image_size)
                    
                    # Combine features: raw RGB + wavelet grayscale
                    raw_features = scaled_raw_img.reshape(image_size[0] * image_size[1] * 3, 1)
                    wavelet_features = scaled_img_wavelet.reshape(image_size[0] * image_size[1], 1)
                    combined_features = np.vstack((raw_features, wavelet_features))
                    
                    X.append(combined_features)
                    y.append(self.class_dict[celebrity_name])
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {training_image}: {e}")
                    continue
        
        # Convert to numpy arrays
        feature_length = image_size[0] * image_size[1] * 3 + image_size[0] * image_size[1]
        self.X = np.array(X).reshape(len(X), feature_length).astype(float)
        self.y = np.array(y)
        
        print(f"Created {self.X.shape[0]} feature vectors with {self.X.shape[1]} features each")
        print(f"Feature composition: {image_size[0]*image_size[1]*3} raw pixels + {image_size[0]*image_size[1]} wavelet pixels")
        
        return self.X, self.y
    
    def train_and_compare_models(self, test_size=0.25, random_state=42):
        """
        Train and compare multiple ML models with hyperparameter tuning.
        
        Args:
            test_size (float): Fraction of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary of trained models and their performance
        """
        print("\n" + "="*60)
        print("TRAINING AND COMPARING MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Define model parameters for grid search
        model_params = {
            'svm': {
                'model': SVC(gamma='auto', probability=True),
                'params': {
                    'svc__C': [1, 10, 100, 1000],
                    'svc__kernel': ['rbf', 'linear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'randomforestclassifier__n_estimators': [1, 5, 10, 20]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(solver='liblinear', multi_class='auto'),
                'params': {
                    'logisticregression__C': [1, 5, 10]
                }
            }
        }
        
        # Train and evaluate models
        scores = []
        best_estimators = {}
        
        for algo, mp in model_params.items():
            print(f"\nTraining {algo}...")
            
            # Create pipeline with scaling
            pipe = make_pipeline(StandardScaler(), mp['model'])
            
            # Grid search with cross-validation
            clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
            clf.fit(X_train, y_train)
            
            # Store results
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
        
        # Create results DataFrame
        results_df = pd.DataFrame(scores)
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Select best model based on test score
        best_model_name = results_df.loc[results_df['test_score'].idxmax(), 'model']
        self.best_model = best_estimators[best_model_name]
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Test accuracy: {results_df['test_score'].max():.4f}")
        
        # Detailed evaluation of best model
        self._evaluate_model(X_test, y_test)
        
        return {
            'results': results_df,
            'best_estimators': best_estimators,
            'best_model': self.best_model,
            'test_data': (X_test, y_test)
        }
    
    def _evaluate_model(self, X_test, y_test):
        """
        Detailed evaluation of the best model.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        """
        print("\n" + "="*60)
        print("DETAILED MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        
        # Classification report
        target_names = list(self.class_dict.keys())
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Sports Celebrity Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path='../saved_artifacts/saved_model.pkl', class_dict_path='../saved_artifacts/class_dictionary.json'):
        """
        Save the trained model and class dictionary.
        
        Args:
            model_path (str): Path to save the model
            class_dict_path (str): Path to save class dictionary
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Run train_and_compare_models() first.")
        
        # Save model
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save class dictionary
        with open(class_dict_path, 'w') as f:
            json.dump(self.class_dict, f, indent=2)
        print(f"Class dictionary saved to: {class_dict_path}")
    
    def load_model(self, model_path='../saved_artifacts/saved_model.pkl', class_dict_path='../saved_artifacts/class_dictionary.json'):
        """
        Load a previously trained model.
        
        Args:
            model_path (str): Path to the saved model
            class_dict_path (str): Path to the class dictionary
        """
        # Load model
        self.best_model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load class dictionary
        with open(class_dict_path, 'r') as f:
            self.class_dict = json.load(f)
        print(f"Class dictionary loaded from: {class_dict_path}")
    
    def predict_celebrity(self, image_path, return_probabilities=False):
        """
        Predict celebrity from a new image.
        
        Args:
            image_path (str): Path to the image
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            str or tuple: Celebrity name or (name, probabilities)
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Train a model or load an existing one.")
        
        # Get cropped face image using flexible detection
        cropped_img = self.data_cleaner.get_cropped_image_with_eye_validation(image_path)
        
        if cropped_img is None:
            return "No face detected"
        
        # Create feature vector
        scaled_raw_img = cv2.resize(cropped_img, (32, 32))
        img_wavelet = self.wavelet.w2d(cropped_img, 'db1', 5)
        scaled_img_wavelet = cv2.resize(img_wavelet, (32, 32))
        
        # Combine features
        raw_features = scaled_raw_img.reshape(32 * 32 * 3, 1)
        wavelet_features = scaled_img_wavelet.reshape(32 * 32, 1)
        combined_features = np.vstack((raw_features, wavelet_features))
        
        # Reshape for prediction
        feature_vector = combined_features.reshape(1, -1).astype(float)
        
        # Make prediction
        prediction = self.best_model.predict(feature_vector)[0]
        celebrity_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(prediction)]
        
        if return_probabilities:
            probabilities = self.best_model.predict_proba(feature_vector)[0]
            prob_dict = {name: prob for name, prob in zip(self.class_dict.keys(), probabilities)}
            return celebrity_name, prob_dict
        
        return celebrity_name
    
    def demonstrate_preprocessing(self, sample_image_path):
        """
        Demonstrate the preprocessing pipeline on a sample image.
        
        Args:
            sample_image_path (str): Path to a sample image
        """
        print("Demonstrating preprocessing pipeline...")
        
        # Load original image
        original_img = cv2.imread(sample_image_path)
        if original_img is None:
            print(f"Could not load image: {sample_image_path}")
            return
        
        # Get cropped face
        cropped_img = self.data_cleaner.get_cropped_image_with_eye_validation(sample_image_path)
        
        if cropped_img is None:
            print("No valid face detected in the image")
            return
        
        # Show preprocessing steps
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Cropped face
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Face (Flexible Detection)')
        plt.axis('off')
        
        # Wavelet transform
        plt.subplot(1, 3, 3)
        wavelet_img = self.wavelet.w2d(cropped_img, 'db1', 5)
        plt.imshow(wavelet_img, cmap='gray')
        plt.title('Wavelet Transform (Edge Features)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the complete sports celebrity classification pipeline."""
    print("Sports Celebrity Image Classification")
    print("Celebrities: Cristiano Ronaldo, Lionel Messi, Steph Curry, Serena Williams, Carlos Alcaraz")
    print()
    
    # Initialize classifier
    classifier = SportsCelebrityClassifier()
    
    # Prepare dataset (or load existing)
    try:
        classifier.load_existing_dataset()
        print("Loaded existing processed dataset")
    except:
        print("Processing raw dataset...")
        classifier.prepare_dataset()
    
    # Create feature vectors
    X, y = classifier.create_feature_vectors()
    
    # Train and compare models
    results = classifier.train_and_compare_models()
    
    # Save the best model
    classifier.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Model and class dictionary saved.")
    print("You can now use the classifier to predict celebrities in new images.")
    
    return classifier, results


if __name__ == "__main__":
    classifier, training_results = main()