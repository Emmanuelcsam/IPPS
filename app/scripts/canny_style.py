"""Processed from canny_style.py - Detected operations: gaussian_blur, sobel_edge, grayscale"""
import cv2
import numpy as np

def process_image(image: np.ndarray, kernel_size: float = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Processed from canny_style.py - Detected operations: gaussian_blur, sobel_edge, grayscale
    
    Args:
        image: Input image
        kernel_size: Kernel size
        sigma: Sigma
    
    Returns:
        Processed image
    """
    try:
        result = image.copy()
        
        # Convert to grayscale if needed
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 1.4)
        
        return result
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return image
