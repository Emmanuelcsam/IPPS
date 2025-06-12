#!/usr/bin/env python3
"""Create cladding mask (excluding core)"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Fiber parameters
center = (image.shape[1]//2, image.shape[0]//2)  # (x, y)
core_radius = 8
cladding_radius = 80

h, w = image.shape[:2]
Y, X = np.ogrid[:h, :w]
dist_sq = (X - center[0])**2 + (Y - center[1])**2

# Create masks
core_mask = (dist_sq <= core_radius**2).astype(np.uint8) * 255
cladding_full = (dist_sq <= cladding_radius**2).astype(np.uint8) * 255

# Cladding = full circle minus core
cladding_mask = cv2.subtract(cladding_full, core_mask)

# Display
cv2.imshow("Cladding Mask", cladding_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()