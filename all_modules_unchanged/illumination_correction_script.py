#!/usr/bin/env python3
"""Correct illumination using morphology"""
import cv2
import numpy as np

# Load image and correct illumination
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Background estimation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Subtract background and shift to mid-gray
corrected = cv2.subtract(image, background)
corrected = cv2.add(corrected, np.full_like(corrected, 128))

cv2.imshow("Original", image)
cv2.imshow("Illumination Corrected", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()