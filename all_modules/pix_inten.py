#!/usr/bin/env python3
"""Get pixel intensity at specific coordinates"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image as grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Failed to load image")
else:
    height, width = img.shape
    
    # Get center pixel intensity
    center_x, center_y = width // 2, height // 2
    center_intensity = img[center_y, center_x]
    
    print(f"Image size: {width}x{height}")
    print(f"Center pixel ({center_x}, {center_y}) intensity: {center_intensity}")
    
    # Get intensity at custom coordinates (example)
    x, y = 100, 100
    if 0 <= x < width and 0 <= y < height:
        intensity = img[y, x]
        print(f"Pixel ({x}, {y}) intensity: {intensity}")
    else:
        print(f"Coordinates ({x}, {y}) are out of bounds")