#!/usr/bin/env python3
"""Adaptive Thresholding"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply different thresholding methods

# 1. Simple global threshold
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 2. Otsu's threshold (automatic threshold selection)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Adaptive threshold - Gaussian weighted mean
adaptive_gaussian = cv2.adaptiveThreshold(image, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, -2)

# 4. Adaptive threshold - Mean
adaptive_mean = cv2.adaptiveThreshold(image, 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, -2)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Global Threshold', global_thresh)
cv2.imshow('Otsu Threshold', otsu_thresh)
cv2.imshow('Adaptive Gaussian', adaptive_gaussian)
cv2.imshow('Adaptive Mean', adaptive_mean)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('global_threshold.png', global_thresh)
cv2.imwrite('otsu_threshold.png', otsu_thresh)
cv2.imwrite('adaptive_gaussian.png', adaptive_gaussian)
cv2.imwrite('adaptive_mean.png', adaptive_mean)