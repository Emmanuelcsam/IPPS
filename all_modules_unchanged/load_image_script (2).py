#!/usr/bin/env python3
"""Load and validate image"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
image = cv2.imread(image_path)

if image is not None:
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    # Save for verification
    cv2.imwrite("loaded_image.jpg", image)
else:
    print("Failed to load image")