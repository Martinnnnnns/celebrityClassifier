'use client';

import { useState, useCallback } from 'react';
import ImageUploader from './components/ImageUploader.js';
import CelebrityShowcase from './components/CelebrityShowcase.js';
import ClassificationResults from './components/ClassificationResults.js';

export default function ClassifierPage() {
  const [classificationResult, setClassificationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageClassification = useCallback(async (imageData) => {
    setIsLoading(true);
    setError(null);
    setClassificationResult(null);

    try {
      const response = await fetch('/api/classify_image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_data: imageData }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Classification failed');
      }

      const result = await response.json();
      
      if (!result || result.length === 0) {
        throw new Error('No face detected in the image. Please try an image showing the face clearly.');
      }

      // Find the best match
      let bestMatch = null;
      let bestScore = -1;
      
      for (const res of result) {
        const maxScore = Math.max(...res.class_probability);
        if (maxScore > bestScore) {
          bestMatch = res;
          bestScore = maxScore;
        }
      }

      setClassificationResult(bestMatch);
    } catch (err) {
      console.error('Classification error:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setClassificationResult(null);
    setError(null);
  }, []);
  const showResultsLayout = classificationResult || isLoading || error;

  return (
    <div className="classifier-main-container">
      <div className="container mx-auto px-4 py-8">
        {/* Sports personalities showcase */}
        <CelebrityShowcase />

        {/* Dynamic upload and results section */}
        <div className={`${showResultsLayout ? 'upload-results-layout' : 'upload-centered-layout'}`}>
          {/* Upload section */}
          <div className={`upload-section-container ${showResultsLayout ? 'upload-left' : 'upload-center'}`}>
            <ImageUploader
              onImageSelect={handleImageClassification}
              onClear={clearResults}
              isLoading={isLoading}
            />
          </div>

          {/* Results section */}
          {showResultsLayout && (
            <div className="results-section-container">
              {error && (
                <div className="error">
                  <p>{error}</p>
                </div>
              )}

              {isLoading && (
                <div className="loading-container text-center p-8">
                  <div className="loading-spinner"></div>
                  <p className="mt-4 text-blue-600">🔄 Processing image with flexible face detection...</p>
                </div>
              )}

              {classificationResult && (
                <ClassificationResults result={classificationResult} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}