#!/usr/bin/env python3
"""Load image and convert to grayscale intensity matrix"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Convert to grayscale (intensity matrix)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save result
cv2.imwrite('intensity_matrix.png', gray)
print(f"Image shape: {gray.shape}")
print(f"Value range: [{gray.min()}, {gray.max()}]")