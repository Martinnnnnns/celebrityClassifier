import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    """
    Apply 2D wavelet transform to extract edge features from an image.

    Args:
        img (numpy.ndarray): Input image in BGR format (as loaded by cv2.imread)
        mode (str): Wavelet mode ('haar', 'db1', etc.)
        level (int): Decomposition level

    Returns:
        numpy.ndarray: Wavelet transformed image emphasizing edges
    """
    imArray = img.copy()

    # Convert BGR to grayscale (OpenCV loads images in BGR, not RGB)
    if len(imArray.shape) == 3:
        imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1] range
    imArray = np.float32(imArray)
    imArray /= 255.0

    # Apply 2D wavelet decomposition
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Zero out approximation coefficients to keep only detail coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruct image from detail coefficients only
    imArray_H = pywt.waverec2(coeffs_H, mode)

    # Convert back to [0, 255] range
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H