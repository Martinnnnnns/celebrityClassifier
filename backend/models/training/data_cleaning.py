"""
Sports Celebrity Image Classification: Improved Data Cleaning

This module handles the preprocessing pipeline for sports celebrity images:
1. Detects faces from images using OpenCV's Haar Cascades
2. Uses more flexible face detection (no strict eye requirement)
3. Multiple detection strategies for better coverage
4. Crops images to focus on the face region
5. Saves processed images to a cropped dataset directory
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil


class ImageDataCleaner:
    """Handles face detection and image cropping for sports celebrity classification."""
    
    def __init__(self, dataset_path="./training_data/", cropped_path="./training_data/cropped/"):
        """
        Initialize the data cleaner with dataset paths.
        
        Args:
            dataset_path (str): Path to the raw dataset directory
            cropped_path (str): Path where cropped images will be saved
        """
        self.dataset_path = dataset_path
        self.cropped_path = cropped_path
        
        # Initialize OpenCV cascade classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier('../../resources/opencv/haarcascades/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('../../resources/opencv/haarcascades/haarcascade_eye.xml')
    
    def get_cropped_image_if_face_detected(self, image_path):
        """
        Detect face in an image and return cropped face.
        Uses multiple detection strategies for better coverage.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray or None: Cropped face image if valid, None otherwise
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Strategy 1: Standard detection
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Strategy 2: More lenient detection if no faces found
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
        
        # Strategy 3: Even more lenient if still no faces
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 2)
        
        if len(faces) > 0:
            # Take the largest face (most likely to be the main subject)
            areas = [w * h for (x, y, w, h) in faces]
            largest_face_idx = np.argmax(areas)
            x, y, w, h = faces[largest_face_idx]
            
            # Add some padding around the face (10% on each side)
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            
            roi_color = img[y_start:y_end, x_start:x_end]
            
            # Basic quality check - ensure the cropped region is not too small
            if roi_color.shape[0] > 30 and roi_color.shape[1] > 30:
                return roi_color
        
        return None
    
    def get_cropped_image_if_2_eyes(self, image_path):
        """
        Original strict method - kept for backward compatibility.
        Detect face and eyes in an image, return cropped face if 2+ eyes detected.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray or None: Cropped face image if valid, None otherwise
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Only return face if 2 or more eyes are detected
            if len(eyes) >= 2:
                return roi_color
        
        return None
    
    def get_cropped_image_with_eye_validation(self, image_path):
        """
        Flexible method - detect face first, then validate with eye detection.
        Falls back to face-only detection if no eyes found.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray or None: Cropped face image if valid, None otherwise
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Try to find faces with eyes first (highest quality)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Prefer faces with 2+ eyes
            if len(eyes) >= 2:
                return roi_color
            # Accept faces with 1 eye as backup
            elif len(eyes) >= 1:
                return roi_color
        
        # Fallback: any detected face
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take first detected face
            return img[y:y+h, x:x+w]
        
        return None
    
    def visualize_face_detection(self, image_path):
        """
        Visualize face and eye detection on an image for debugging.
        
        Args:
            image_path (str): Path to the input image
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        cv2.destroyAllWindows()
        face_img = img.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = face_img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Face and Eye Detection - Found {len(faces)} faces')
        plt.axis('off')
        plt.show()
    
    def get_image_directories(self):
        """
        Get all subdirectories in the dataset path, excluding the cropped folder.
        
        Returns:
            list: List of directory paths containing celebrity images
        """
        img_dirs = []
        for entry in os.scandir(self.dataset_path):
            if entry.is_dir() and entry.name != 'cropped':  # Exclude cropped folder
                img_dirs.append(entry.path)
        return img_dirs
    
    def setup_cropped_directory(self):
        """Setup the cropped images directory, removing existing one if present."""
        if os.path.exists(self.cropped_path):
            shutil.rmtree(self.cropped_path)
        os.makedirs(self.cropped_path)
    
    def process_all_images(self, detection_method='flexible'):
        """
        Process all images in the dataset: detect faces, crop, and save valid images.
        
        Args:
            detection_method (str): 'flexible', 'strict', or 'face_only'
                - 'flexible': Try eye validation, fall back to face-only
                - 'strict': Original method (2+ eyes required)
                - 'face_only': Just face detection, no eye requirement
        
        Returns:
            tuple: (list of cropped directories, dict of celebrity file names)
        """
        # Get all celebrity directories
        img_dirs = self.get_image_directories()
        print(f"Found celebrity directories: {[d.split('/')[-1] for d in img_dirs]}")
        print(f"Using detection method: {detection_method}")
        
        # Setup output directory
        self.setup_cropped_directory()
        
        # Select detection method
        if detection_method == 'strict':
            detection_func = self.get_cropped_image_if_2_eyes
        elif detection_method == 'face_only':
            detection_func = self.get_cropped_image_if_face_detected
        else:  # flexible
            detection_func = self.get_cropped_image_with_eye_validation
        
        cropped_image_dirs = []
        celebrity_file_names_dict = {}
        total_processed = 0
        total_failed = 0
        
        for img_dir in img_dirs:
            count = 1
            failed_count = 0
            celebrity_name = img_dir.split('/')[-1]
            print(f"\nProcessing {celebrity_name}...")
            
            celebrity_file_names_dict[celebrity_name] = []
            
            # Process each image in the celebrity directory
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    roi_color = detection_func(entry.path)
                    
                    if roi_color is not None:
                        # Create celebrity folder in cropped directory
                        cropped_folder = os.path.join(self.cropped_path, celebrity_name)
                        if not os.path.exists(cropped_folder):
                            os.makedirs(cropped_folder)
                            cropped_image_dirs.append(cropped_folder)
                            print(f"Created cropped images folder: {cropped_folder}")
                        
                        # Save cropped image
                        cropped_file_name = f"{celebrity_name}{count}.png"
                        cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
                        
                        cv2.imwrite(cropped_file_path, roi_color)
                        celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                        count += 1
                        total_processed += 1
                    else:
                        failed_count += 1
                        total_failed += 1
            
            success_rate = (count-1) / ((count-1) + failed_count) * 100 if (count-1) + failed_count > 0 else 0
            print(f"Processed {count-1} valid images for {celebrity_name} (Success rate: {success_rate:.1f}%)")
            if failed_count > 0:
                print(f"  Failed to process {failed_count} images")
        
        overall_success_rate = total_processed / (total_processed + total_failed) * 100 if (total_processed + total_failed) > 0 else 0
        print(f"\nOverall success rate: {overall_success_rate:.1f}% ({total_processed}/{total_processed + total_failed})")
        
        return cropped_image_dirs, celebrity_file_names_dict
    
    def get_processing_stats(self, celebrity_file_names_dict):
        """
        Print statistics about the processed dataset.
        
        Args:
            celebrity_file_names_dict (dict): Dictionary mapping celebrities to their processed files
        """
        print("\n" + "="*50)
        print("DATASET PROCESSING SUMMARY")
        print("="*50)
        
        total_images = 0
        for celebrity, files in celebrity_file_names_dict.items():
            count = len(files)
            total_images += count
            print(f"{celebrity}: {count} images")
        
        print(f"\nTotal processed images: {total_images}")
        print("="*50)


def main():
    """Main function to run the data cleaning pipeline."""
    print("Starting Sports Celebrity Image Classification - Improved Data Cleaning")
    print("Celebrities: Cristiano Ronaldo, Lionel Messi, Steph Curry, Serena Williams, Carlos Alcaraz")
    print("\nAvailable detection methods:")
    print("  'flexible' - Try eye validation, fall back to face-only (RECOMMENDED)")
    print("  'face_only' - Just face detection, no eye requirement")  
    print("  'strict' - Original method (2+ eyes required)")
    
    # Initialize the data cleaner
    cleaner = ImageDataCleaner()
    
    # Use flexible detection method for best results
    detection_method = 'flexible'  # Change this to 'face_only' or 'strict' if needed
    
    # Process all images
    cropped_dirs, celebrity_files = cleaner.process_all_images(detection_method=detection_method)
    
    # Show processing statistics
    cleaner.get_processing_stats(celebrity_files)
    
    print(f"\nData cleaning completed successfully using '{detection_method}' method!")
    print("If you're not satisfied with the results, try:")
    print("  - 'face_only' method for maximum coverage")
    print("  - 'strict' method for highest quality (original)")
    
    return cropped_dirs, celebrity_files


if __name__ == "__main__":
    cropped_directories, celebrity_file_names = main()