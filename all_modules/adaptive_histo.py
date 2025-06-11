#!/usr/bin/env python3
"""Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Apply CLAHE
enhanced = clahe.apply(img)

cv2.imwrite('clahe_enhanced.png', enhanced)