#!/usr/bin/env python3
"""
Image Loading and Preprocessing Module
Handles image loading, grayscale conversion, and basic preprocessing
"""
import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """Load image from file path"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def preprocess_image(image, clahe_clip=2.0, blur_kernel=5):
    """
    Preprocess image with CLAHE and Gaussian blur
    Returns: grayscale, preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply CLAHE for illumination correction
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
    
    return gray, blurred


def correct_illumination(image):
    """Simple illumination correction using morphological operations"""
    # Background estimation using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Subtract background
    corrected = cv2.subtract(image, background)
    corrected = cv2.add(corrected, np.full_like(corrected, 128))  # Shift to mid-gray
    
    return corrected


# Test function
if __name__ == "__main__":
    # Create test image
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 80, (128, 128, 128), -1)
    cv2.imwrite("test_fiber.png", test_img)
    
    # Test loading and preprocessing
    loaded = load_image("test_fiber.png")
    gray, processed = preprocess_image(loaded)
    print(f"Loaded image shape: {loaded.shape}")
    print(f"Processed image shape: {processed.shape}")
    
    # Cleanup
    Path("test_fiber.png").unlink()
