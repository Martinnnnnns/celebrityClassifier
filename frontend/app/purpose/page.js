export default function PurposePage() {
  return (
    <div className="main-container">
      <div className="purpose-main-container">
        <h1 className="purpose-title">Project Purpose</h1>
        
        <div className="purpose-main-card">
          <div className="purpose-sections">
            <section className="purpose-section">
              <h2>The Vision</h2>
              <p>
                The Sports Celebrity Classifier demonstrates a complete end-to-end machine learning pipeline for image recognition. 
                This project showcases how advanced computer vision and AI can identify five world-renowned athletes: 
                Cristiano Ronaldo, Lionel Messi, Steph Curry, Serena Williams, and Carlos Alcaraz. Its important to note
                that due to the limitation of images per celebrity(~150 preprocessing), the model will give wrong results.
                I am currently working on scraping images to have a more robust dataset.
              </p>
            </section>
            
            <section className="purpose-section">
              <h2>Technical Pipeline</h2>
              <ol className="purpose-ordered-list list-decimal">
                <li>
                  <strong>Data Preprocessing:</strong> Intelligent face detection using OpenCV Haar Cascades with flexible detection strategies and automatic cropping
                </li>
                <li>
                  <strong>Feature Engineering:</strong> Combines raw pixel data with wavelet transform processing to extract enhanced facial features
                </li>
                <li>
                  <strong>Model Training:</strong> Compares SVM, Random Forest, and Logistic Regression with grid search optimization
                </li>
                <li>
                  <strong>Model Deployment:</strong> Best-performing model serialized to .pkl file for real-time predictions
                </li>
                <li>
                  <strong>Web Interface:</strong> Next.js frontend with drag-and-drop upload and live classification results
                </li>
              </ol>
            </section>
            
            <section className="purpose-section">
              <h2>Key Innovations</h2>
              <ul className="purpose-list list-disc">
                <li><strong>Flexible Face Detection</strong> - Multi-strategy approach with eye validation fallback for maximum coverage</li>
                <li><strong>Wavelet Transform Features</strong> - Enhanced edge detection and texture analysis for better discrimination</li>
                <li><strong>Multi-Model Comparison</strong> - Automated hyperparameter tuning to select optimal algorithm</li>
                <li><strong>Real-Time Classification</strong> - Instant predictions with confidence scores and probability rankings</li>
              </ul>
            </section>
            
            <section className="purpose-section">
              <h2>Technologies</h2>
              <ul className="purpose-list list-disc">
                <li><strong>Computer Vision:</strong> OpenCV, Haar Cascades, Wavelet Transforms</li>
                <li><strong>Machine Learning:</strong> Scikit-learn, SVM, Random Forest, Grid Search CV</li>
                <li><strong>Backend:</strong> Python, NumPy, Pandas, Joblib</li>
                <li><strong>Frontend:</strong> Next.js, React, JavaScript, Drag-and-Drop API</li>
              </ul>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}