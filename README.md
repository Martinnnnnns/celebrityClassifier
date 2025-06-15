# Sports Celebrity Image Classification

A machine learning-powered web application that classifies images of sports celebrities using computer vision and deep learning techniques. The system can identify five prominent sports figures: **Cristiano Ronaldo**, **Lionel Messi**, **Steph Curry**, **Serena Williams**, and **Carlos Alcaraz**.

## Features

- **Real-Time Classification**: Upload an image and get instant celebrity identification with confidence scores
- **Advanced Face Detection**: OpenCV Haar Cascades with flexible detection strategies (multiple fallback methods)
- **Wavelet Transform**: Combines raw pixel data with wavelet-transformed features for enhanced accuracy
- **Multi-Model Comparison**: Automatically compares SVM, Random Forest, and Logistic Regression with hyperparameter tuning
- **Modern Web Interface**: 
  - React-based Next.js frontend with smooth animations
  - Drag-and-drop image upload with preview
  - Responsive design that works on all devices
  - Real-time results with probability rankings
- **Professional UI/UX**: 
  - Celebrity showcase with high-quality images
  - Interactive navigation between multiple pages
  - Smooth animations and transitions
  - Clean, modern design system
- **Comprehensive Results**: 
  - Winner card with highest confidence celebrity
  - Complete probability table for all celebrities
  - Visual feedback and error handling
- **Robust Processing**: Handles various image formats, sizes, and quality levels automatically

## Architecture

### Backend (Flask)
- **Image Processing**: OpenCV for face detection and cropping
- **Feature Extraction**: Combines raw pixels with wavelet transform features
- **Machine Learning**: Scikit-learn models with grid search optimization
- **API Endpoints**: RESTful services for classification, health checks, and metadata

### Frontend (Next.js)
- **Modern React**: TypeScript-based Next.js application
- **Responsive UI**: Tailwind CSS for styling and mobile-first design
- **File Upload**: React Dropzone for intuitive image selection
- **Real-time Results**: Instant classification with probability scores

### Machine Learning Pipeline
1. **Data Cleaning**: Flexible face detection with multiple fallback strategies
2. **Feature Engineering**: Raw pixel data + wavelet transform for edge detection
3. **Model Training**: Grid search across multiple algorithms
4. **Model Selection**: Automated best model selection based on performance
5. **Deployment**: Joblib model serialization for production use

## Quick Start

### Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **Required Python packages**: OpenCV, NumPy, Scikit-learn, Flask, etc.

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend  # or wherever your Python files are located
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install flask flask-cors opencv-python numpy joblib scikit-learn pywt matplotlib pandas seaborn
   ```

4. **Ensure you have the trained model files:**
   ```
   models/saved_artifacts/
   ├── saved_model.pkl          # Pre-trained scikit-learn model
   └── class_dictionary.json    # Celebrity class mappings
   ```
   
   *Note: If you don't have these files, you'll need to run the training pipeline first (see [Training Your Own Model](#training-your-own-model)).*

5. **Verify OpenCV cascade files exist:**
   ```
   resources/opencv/haarcascades/
   ├── haarcascade_frontalface_default.xml
   └── haarcascade_eye.xml
   ```

6. **Start the Flask backend:**
   ```bash
   python app.py
   ```
   
   The backend will start on `http://127.0.0.1:5000` with detailed startup information.

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend  # or cd app if using app router structure
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   # This installs: next, react, react-dom, react-dropzone, react-icons, tailwindcss, etc.
   ```

3. **Ensure you have the celebrity images:**
   ```
   public/images/
   ├── ronaldo.jpeg
   ├── messi.jpeg
   ├── curry.jpeg
   ├── serena.jpeg
   ├── alcaraz.jpeg
   └── upload.png
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```
   
   The frontend will start on `http://localhost:3000`

### Accessing the Application

Once both servers are running:
- **Main App**: Open `http://localhost:3000` in your browser
- **Classifier**: Main page with celebrity showcase and image upload
- **About**: Personal information and background (`/about`)
- **Purpose**: Project goals and technical details (`/purpose`)
- **Contacts**: Contact information (`/contacts`)
- **Other Work**: Portfolio showcase (`/other-work`)
- **Backend API**: Available at `http://127.0.0.1:5000/api/`
- **Health Check**: `http://127.0.0.1:5000/api/health`

## Project Structure

```
sports-celebrity-classifier/
├── backend/
│   ├── app.py                      # Flask API server with CORS and error handling
│   ├── image_utils.py              # Image processing and classification logic
│   ├── wavelet_transform.py        # Wavelet feature extraction utilities
│   ├── celebrity_classifier.py     # Complete ML training pipeline
│   ├── data_cleaning.py           # Flexible face detection and data preprocessing
│   └── models/
│       └── saved_artifacts/
│           ├── saved_model.pkl     # Trained scikit-learn model
│           └── class_dictionary.json # Class mappings
├── frontend/
│   ├── app/
│   │   ├── layout.js              # Root layout with navigation
│   │   ├── page.js                # Main classifier page
│   │   ├── globals.css            # Global styles and animations
│   │   ├── about/page.js          # About page
│   │   ├── purpose/page.js        # Project purpose page
│   │   ├── contacts/page.js       # Contact information
│   │   ├── other-work/page.js     # Portfolio showcase
│   │   └── components/
│   │       ├── Navigation.js      # Site navigation component
│   │       ├── CelebrityShowcase.js # Celebrity display grid
│   │       ├── ImageUploader.js   # Drag-and-drop upload component
│   │       └── ClassificationResults.js # Results display
│   ├── public/
│   │   └── images/               # Celebrity photos and assets
│   ├── package.json
│   └── next.config.js            # Next.js configuration with API proxy
├── resources/
│   └── opencv/
│       └── haarcascades/          # OpenCV cascade files
└── training_data/                 # Dataset (if training from scratch)
    ├── cristiano_ronaldo/
    ├── lionel_messi/
    ├── steph_curry/
    ├── serena_williams/
    └── carlos_alcaraz/
```

## Training Your Own Model

If you want to train the model from scratch:

1. **Prepare your dataset:**
   - Create folders for each celebrity in `training_data/`
   - Add images (JPG, PNG) to respective folders
   - Recommended: 100+ images per celebrity

2. **Run data cleaning:**
   ```bash
   python data_cleaning.py
   ```

3. **Train the model:**
   ```bash
   python celebrity_classifier.py
   ```

4. **The trained model will be saved to `models/saved_artifacts/`**

## User Interface Features

### Main Classifier Page (`/`)
- **Celebrity Showcase**: Grid display of all 5 celebrities with professional styling
- **Dynamic Layout**: Switches between centered upload (no results) and two-column layout (with results)
- **Image Upload**: 
  - Drag-and-drop interface with visual feedback
  - Image compression and preprocessing
  - File size validation and error handling
  - Preview with circular cropping
- **Results Display**:
  - Winner card with celebrity image and confidence score
  - Complete probability table ranked by confidence
  - Smooth animations and transitions

### Additional Pages
- **About Page (`/about`)**: Personal background with profile image and skills
- **Purpose Page (`/purpose`)**: Technical project details and methodology
- **Contacts Page (`/contacts`)**: Professional contact information
- **Other Work Page (`/other-work`)**: Portfolio showcase with project descriptions

### Design System
- **Color Palette**: Professional grays with green accents for success states
- **Typography**: Segoe UI font family with consistent sizing hierarchy
- **Animations**: Slide-in effects, hover transitions, and loading states
- **Responsive**: Mobile-first design that adapts to all screen sizes

## API Endpoints

### Classification
```http
POST /api/classify_image
Content-Type: application/json

{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

**Response:**
```json
{
  "class": "cristiano_ronaldo",
  "class_probability": [15.2, 78.9, 3.1, 1.8, 1.0],
  "class_dictionary": {
    "cristiano_ronaldo": 0,
    "lionel_messi": 1,
    "steph_curry": 2,
    "serena_williams": 3,
    "carlos_alcaraz": 4
  },
  "detection_info": {
    "method": "flexible_detection",
    "face_number": 1,
    "total_faces": 1
  }
}
```
## Model Performance

The system uses a multi-model approach with grid search optimization:

- **SVM**: Best for complex decision boundaries
- **Random Forest**: Ensemble method for robustness
- **Logistic Regression**: Fast and interpretable baseline

Features:
- **Raw Pixels**: 32x32x3 = 3,072 features
- **Wavelet Transform**: 32x32 = 1,024 edge features
- **Total**: 4,096 features per image

## Technology Stack

### Backend
- **Flask**: Web framework
- **OpenCV**: Computer vision and face detection
- **Scikit-learn**: Machine learning models
- **NumPy**: Numerical computing
- **PyWavelets**: Wavelet transform
- **Joblib**: Model serialization

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript development
- **React Dropzone**: Advanced file upload with drag-and-drop
- **React Icons**: Comprehensive icon library
- **CSS Animations**: Custom keyframe animations for smooth UX
- **Responsive Design**: Mobile-first approach with breakpoint optimization

