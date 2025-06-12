#!/usr/bin/env python3
"""Basic Morphological Operations (Dilation and Erosion)"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create morphological kernel
kernel_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Apply dilation (max filter)
dilated = cv2.dilate(image, kernel)

# Apply erosion (min filter)
eroded = cv2.erode(image, kernel)

# Calculate morphological gradient (difference)
gradient = cv2.subtract(dilated, eroded)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Dilated', dilated)
cv2.imshow('Eroded', eroded)
cv2.imshow('Morphological Gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save results
cv2.imwrite('dilated.png', dilated)
cv2.imwrite('eroded.png', eroded)
cv2.imwrite('morphological_gradient.png', gradient)