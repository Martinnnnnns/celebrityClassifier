'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export default function ImageUploader({ onImageSelect, onClear, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [compressedDataUrl, setCompressedDataUrl] = useState(null);

  const compressImage = useCallback((file) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        const fileSizeMB = file.size / (1024 * 1024);
        let { width, height } = img;

        let maxDimension = 800;
        if (fileSizeMB > 4) maxDimension = 600;
        if (fileSizeMB > 6) maxDimension = 500;
        if (fileSizeMB > 8) maxDimension = 400;

        // Resize if needed
        if (width > maxDimension || height > maxDimension) {
          if (width > height) {
            height = (height * maxDimension) / width;
            width = maxDimension;
          } else {
            width = (width * maxDimension) / height;
            height = maxDimension;
          }
        }

        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        // Progressive quality reduction
        let quality = 0.8;
        if (fileSizeMB > 4) quality = 0.7;
        if (fileSizeMB > 6) quality = 0.6;
        if (fileSizeMB > 8) quality = 0.5;

        const compressedDataUrl = canvas.toDataURL('image/jpeg', quality);
        resolve(compressedDataUrl);
      };

      img.src = URL.createObjectURL(file);
    });
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > 8) {
      alert('Image file is too large. Please use an image smaller than 8MB.');
      return;
    }

    setSelectedFile(file);
    
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);

    try {
      const compressedDataUrl = await compressImage(file);
      setCompressedDataUrl(compressedDataUrl);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing image. Please try again.');
    }
  }, [compressImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    maxFiles: 1,
    multiple: false
  });

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setCompressedDataUrl(null);
    onClear();
  };

  const handleClassify = () => {
    if (compressedDataUrl && onImageSelect) {
      onImageSelect(compressedDataUrl);
    }
  };

  return (
    <div className="upload-section">
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="dz-message">
            <img src="/images/upload.png" alt="Upload" width={50} height={50} />
            <br />
            <span className="note">
              {isDragActive ? 'Drop the image here...' : 'Drop files here or click to upload'}
            </span>
          </div>
        </div>
      ) : (
        <div className="file-preview">
          {preview && (
            <div className="preview-image">
              <img src={preview} alt="Preview" />
            </div>
          )}
          <div className="file-details">
            <div className="filename">{selectedFile.name}</div>
            <div className="filesize">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </div>
          </div>
          <button
            onClick={handleClear}
            className="remove-button"
            disabled={isLoading}
          >
            Remove file
          </button>
        </div>
      )}

      {selectedFile && compressedDataUrl && (
        <div className="action-buttons">
          <button
            onClick={handleClassify}
            disabled={isLoading}
            className="classify-button"
          >
            {isLoading ? 'Classifying...' : 'Classify'}
          </button>
        </div>
      )}
    </div>
  );
}