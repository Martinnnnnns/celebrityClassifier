from flask import Flask, request, jsonify
from flask_cors import CORS
import image_utils
import os
import werkzeug.formparser

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  
werkzeug.formparser.max_form_memory_size = 16 * 1024 * 1024  

@app.route('/api/classify_image', methods=['POST', 'OPTIONS'])
def classify_image():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        print("=== Classify image request received ===")
        print("Request method:", request.method)
        print("Content length:", request.content_length)
        print("Content type:", request.content_type)
        
        # Handle both form data and JSON requests
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if not data or 'image_data' not in data:
                return jsonify({"error": "image_data parameter missing"}), 400
            image_data = data['image_data']
        else:
            # Handle form data (backwards compatibility)
            print("Form data keys:", list(request.form.keys()))
            if 'image_data' not in request.form:
                print("ERROR: image_data not found in form")
                return jsonify({"error": "image_data parameter missing"}), 400
            image_data = request.form['image_data']
        
        print("Image data length:", len(image_data))
        
        if image_data.startswith('data:'):
            header, encoded = image_data.split(',', 1)
            actual_size_bytes = len(encoded) * 3 // 4
        else:
            actual_size_bytes = len(image_data) * 3 // 4
            
        actual_size_mb = actual_size_bytes / (1024 * 1024)
        print(f"Actual image size: {actual_size_mb:.2f} MB")
        
        print("Starting classification with flexible face detection...")
        
        # Try to classify the image using flexible detection
        result = image_utils.classify_image(image_data)
        print("Classification result:", result)
        
        # Check if face detection failed (empty result)
        if not result or len(result) == 0:
            return jsonify({
                "error": "No face detected in the image. Please ensure the image clearly shows a person's face. The system now uses flexible detection that should work with most face angles and lighting conditions.",
                "error_type": "no_face_detected",
                "suggestions": [
                    "Make sure the person's face is clearly visible",
                    "Ensure good lighting on the face",
                    "Try with a different angle or photo",
                    "The image should show at least the person's face area"
                ]
            }), 400
        
        # Check if any valid classifications were made
        valid_results = [r for r in result if r and 'class' in r]
        if not valid_results:
            return jsonify({
                "error": "Could not classify the face in the image. The face was detected but classification failed.",
                "error_type": "classification_failed",
                "suggestions": [
                    "Try with a clearer, higher resolution photo",
                    "Ensure the lighting is good on the person's face",
                    "Make sure the face is not too blurry or distorted",
                    "The system works best with front-facing portraits"
                ]
            }), 400
        
        for i, res in enumerate(valid_results):
            res['detection_info'] = {
                'method': 'flexible_detection',
                'face_number': i + 1,
                'total_faces': len(valid_results)
            }
        
        return jsonify(valid_results)
        
    except Exception as e:
        print("=== ERROR in classify_image ===")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        error_message = str(e)
        
        # Handle specific error types with more detailed messages
        if "413" in error_message or "too large" in error_message.lower() or "capacity limit" in error_message.lower():
            return jsonify({
                "error": "Image file is too large. Please use a smaller image (under 5MB).",
                "error_type": "file_too_large",
                "suggestions": [
                    "Resize the image to a smaller resolution",
                    "Compress the image quality",
                    "Use a different image format (JPG is usually smaller than PNG)"
                ]
            }), 413
        elif "timeout" in error_message.lower():
            return jsonify({
                "error": "Request timed out. Please try with a smaller image.",
                "error_type": "timeout",
                "suggestions": [
                    "Try with a smaller image file",
                    "Check your internet connection",
                    "Retry the request"
                ]
            }), 408
        elif "could not load image" in error_message.lower() or "error decoding" in error_message.lower():
            return jsonify({
                "error": "Invalid image format. Please use a valid image file (JPG, PNG, etc.).",
                "error_type": "invalid_image",
                "suggestions": [
                    "Make sure the file is a valid image (JPG, PNG, GIF, etc.)",
                    "Try with a different image",
                    "Check if the image file is corrupted"
                ]
            }), 400
        elif "cascade" in error_message.lower() or "opencv" in error_message.lower():
            return jsonify({
                "error": "Face detection system error. Please try again.",
                "error_type": "detection_system_error",
                "suggestions": [
                    "This might be a temporary system issue",
                    "Try again in a few moments",
                    "If the problem persists, contact support"
                ]
            }), 500
        else:
            return jsonify({
                "error": f"Classification failed: {error_message}",
                "error_type": "general_error",
                "suggestions": [
                    "Try with a different image",
                    "Make sure the image clearly shows a person's face",
                    "Check that the image is not corrupted"
                ]
            }), 500

@app.errorhandler(413)
def too_large(e):
    print("=== 413 error handler triggered ===")
    print("Error details:", str(e))
    return jsonify({
        "error": "File too large. Please use an image smaller than 5MB.",
        "error_type": "file_too_large",
        "suggestions": [
            "Resize the image to a smaller resolution",
            "Compress the image quality",
            "Use JPG format instead of PNG for smaller file size"
        ]
    }), 413

@app.before_request
def log_request_info():
    if request.endpoint == 'classify_image' and request.method == 'POST':
        print(f"=== Before request processing ===")
        print(f"Content-Length header: {request.headers.get('Content-Length', 'Not set')}")
        print(f"Content-Type header: {request.headers.get('Content-Type', 'Not set')}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running"""
    try:
        if image_utils.__model is None:
            return jsonify({
                "status": "unhealthy", 
                "message": "Model not loaded"
            }), 500
        
        return jsonify({
            "status": "healthy",
            "message": "Sports Celebrity Classification service is running",
            "detection_method": "flexible",
            "celebrities": list(image_utils.__class_name_to_number.keys()) if image_utils.__class_name_to_number else []
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }), 500

@app.route('/api/celebrities', methods=['GET'])
def get_celebrities():
    """Get list of celebrities that can be classified"""
    try:
        if not image_utils.__class_name_to_number:
            return jsonify({"error": "Classification system not initialized"}), 500
        
        celebrities = list(image_utils.__class_name_to_number.keys())
        return jsonify({
            "celebrities": celebrities,
            "total_count": len(celebrities),
            "detection_method": "flexible"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get celebrities: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Python Flask API Server For Sports Celebrity Image Classification")
    print("=" * 70)
    print("Configuration:")
    print(f"  API Mode: True")
    print(f"  CORS Enabled: http://localhost:3000")
    print(f"  Maximum content size: 32MB")
    print(f"  Maximum form data size: 16MB")
    print(f"  Detection method: Flexible (multiple strategies)")
    print(f"  Celebrities: Cristiano Ronaldo, Lionel Messi, Steph Curry, Serena Williams, Carlos Alcaraz")
    print("=" * 70)
    
    try:
        image_utils.load_saved_artifacts()
        print("✓ Model and artifacts loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load artifacts: {e}")
        print("Make sure the model has been trained and saved first!")
    
    print("Starting API server on http://127.0.0.1:5000")
    print("Available API endpoints:")
    print("  POST /api/classify_image - Image classification")
    print("  GET  /api/health        - Health check")
    print("  GET  /api/celebrities   - List available celebrities")
    print("Frontend should run on: http://localhost:3000")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)