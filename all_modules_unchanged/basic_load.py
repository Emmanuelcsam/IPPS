#!/usr/bin/env python3
"""Load image and display basic information"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"Failed to load image: {image_path}")
else:
    print(f"Image loaded successfully")
    print(f"Shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Channels: {1 if len(img.shape) == 2 else img.shape[2]}")
    print(f"Size: {img.shape[0]} x {img.shape[1]} pixels")