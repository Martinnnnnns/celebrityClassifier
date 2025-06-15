import joblib
import json
import numpy as np
import base64
import cv2
from wavelet_transform import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    """
    Classify sports celebrity from image using flexible face detection.
    
    Args:
        image_base64_data (str): Base64 encoded image data
        file_path (str): Optional file path for image
        
    Returns:
        list: Classification results with probabilities
    """
    # Use flexible face detection (matching the improved data cleaning logic)
    imgs = get_cropped_image_flexible(file_path, image_base64_data)

    result = []
    for img in imgs:
        try:
            # Resize and process image (same as training pipeline)
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

            len_image_array = 32*32*3 + 32*32

            final = combined_img.reshape(1, len_image_array).astype(float)
            
            # Make prediction
            prediction = __model.predict(final)[0]
            probabilities = __model.predict_proba(final)[0]
            
            result.append({
                'class': class_number_to_name(prediction),
                'class_probability': np.around(probabilities*100, 2).tolist(),
                'class_dictionary': __class_name_to_number
            })
            
        except Exception as e:
            print(f"Error processing face crop: {e}")
            continue

    return result

def class_number_to_name(class_num):
    """Convert class number to celebrity name."""
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    """Load saved model and class dictionary."""
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    class_dict_path = "../models/saved_artifacts/class_dictionary.json"
    model_path = "../models/saved_artifacts/saved_model.pkl"
    
    try:
        with open(class_dict_path, "r") as f:
            __class_name_to_number = json.load(f)
            __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
        print(f"✓ Class dictionary loaded from: {class_dict_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Class dictionary not found at: {class_dict_path}")

    global __model
    if __model is None:
        try:
            with open(model_path, 'rb') as f:
                __model = joblib.load(f)
            print(f"✓ Model loaded from: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    """
    Convert base64 string to OpenCV image.
    
    Args:
        b64str (str): Base64 encoded image string
        
    Returns:
        numpy.ndarray: OpenCV image array
    """
    try:
        encoded_data = b64str.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def get_cropped_image_flexible(image_path, image_base64_data):
    """
    Flexible face detection method that matches the improved data cleaning logic.
    Uses multiple strategies to maximize face detection success rate.
    
    Args:
        image_path (str): Path to image file (optional)
        image_base64_data (str): Base64 encoded image data (optional)
        
    Returns:
        list: List of cropped face images
    """
    face_cascade = cv2.CascadeClassifier('../resources/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../resources/opencv/haarcascades/haarcascade_eye.xml')

    # Load image from file or base64
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        print("ERROR: Could not load image")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped_faces = []
    
    print("Starting flexible face detection...")
    
    # Strategy 1: Try to find faces with 2+ eyes (highest quality)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"Strategy 1 - Standard detection (2+ eyes): Found {len(faces)} faces")
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(f"  Face at ({x},{y},{w},{h}) has {len(eyes)} eyes")
        if len(eyes) >= 2:
            # Add padding around face
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            padded_face = img[y_start:y_end, x_start:x_end]
            cropped_faces.append(padded_face)
    
    if cropped_faces:
        print(f"SUCCESS: Found {len(cropped_faces)} high-quality faces (2+ eyes)")
        return cropped_faces
    
    # Strategy 2: Try to find faces with 1+ eyes (medium quality)
    print("Strategy 2 - Relaxed eye requirement (1+ eye)")
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 1:
            # Add padding around face
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            padded_face = img[y_start:y_end, x_start:x_end]
            cropped_faces.append(padded_face)
    
    if cropped_faces:
        print(f"SUCCESS: Found {len(cropped_faces)} medium-quality faces (1+ eye)")
        return cropped_faces
    
    # Strategy 3: More lenient face detection parameters
    print("Strategy 3 - More lenient face detection")
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    print(f"  Lenient detection: Found {len(faces)} faces")
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 1:
            # Add padding around face
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            padded_face = img[y_start:y_end, x_start:x_end]
            cropped_faces.append(padded_face)
    
    if cropped_faces:
        print(f"SUCCESS: Found {len(cropped_faces)} faces with lenient detection")
        return cropped_faces
    
    # Strategy 4: Even more lenient face detection
    print("Strategy 4 - Very lenient face detection")
    faces = face_cascade.detectMultiScale(gray, 1.05, 2)
    print(f"  Very lenient detection: Found {len(faces)} faces")
    
    if len(faces) > 0:
        # Take the largest face (most likely to be the main subject)
        areas = [w * h for (x, y, w, h) in faces]
        largest_face_idx = np.argmax(areas)
        x, y, w, h = faces[largest_face_idx]
        
        # Basic quality check
        if w > 30 and h > 30:
            # Add padding around face
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            padded_face = img[y_start:y_end, x_start:x_end]
            cropped_faces.append(padded_face)
            print(f"SUCCESS: Using largest face (no eye requirement)")
            return cropped_faces
    
    print("FAILED: No faces detected with any strategy")
    return []

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Original strict method - kept for backward compatibility.
    Only returns faces with 2+ detected eyes.
    
    Args:
        image_path (str): Path to image file (optional)
        image_base64_data (str): Base64 encoded image data (optional)
        
    Returns:
        list: List of cropped face images with 2+ eyes
    """
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        print("ERROR: Could not load image")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cropped_faces = []
    print(f"Original strict method: Found {len(faces)} faces")
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(f"  Face at ({x},{y},{w},{h}) has {len(eyes)} eyes")
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    if cropped_faces:
        print(f"SUCCESS: Found {len(cropped_faces)} faces with 2+ eyes")
    else:
        print("FAILED: No faces with 2+ eyes detected")
    
    return cropped_faces

if __name__ == '__main__':
    load_saved_artifacts()

    # Test with sample images
    # print(classify_image(None, "./test_images/ronaldo.jpeg"))
    # print(classify_image(None, "./test_images/messi.jpeg"))
    # print(classify_image(None, "./test_images/curry.jpeg"))
    # print(classify_image(None, "./test_images/serena.jpeg"))
    # print(classify_image(None, "./test_images/alcaraz.jpeg"))