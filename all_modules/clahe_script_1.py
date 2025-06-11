#!/usr/bin/env python3
"""CLAHE (Contrast Limited Adaptive Histogram Equalization) Enhancement"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Apply CLAHE
enhanced = clahe.apply(image)

# Also show regular histogram equalization for comparison
equalized = cv2.equalizeHist(image)

# Display results
cv2.imshow('Original', image)
cv2.imshow('CLAHE Enhanced', enhanced)
cv2.imshow('Regular Histogram Equalization', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('clahe_enhanced.png', enhanced)
cv2.imwrite('histogram_equalized.png', equalized)