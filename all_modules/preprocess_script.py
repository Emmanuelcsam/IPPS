#!/usr/bin/env python3
"""Preprocess image for better fiber detection"""
import cv2
import numpy as np

# File path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized = cv2.equalizeHist(gray)

# Apply denoising
denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)

# Apply sharpening kernel
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(denoised, -1, kernel)

# Save preprocessed image
cv2.imwrite("preprocessed.jpg", sharpened)
print("Preprocessing complete - saved as preprocessed.jpg")