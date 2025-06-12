#!/usr/bin/env python3
"""Visualize fiber zones combined"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Parameters
center = (image.shape[1]//2, image.shape[0]//2)
core_radius = 8
cladding_radius = 80

h, w = image.shape[:2]
Y, X = np.ogrid[:h, :w]
dist_sq = (X - center[0])**2 + (Y - center[1])**2

# Create masks
core_mask = (dist_sq <= core_radius**2).astype(np.uint8) * 255
cladding_full = (dist_sq <= cladding_radius**2).astype(np.uint8) * 255
cladding_mask = cv2.subtract(cladding_full, core_mask)

# Combine with different intensities
combined = np.zeros((h, w), dtype=np.uint8)
combined = cv2.add(combined, core_mask // 2)  # Core at 50% intensity
combined = cv2.add(combined, cladding_mask // 4)  # Cladding at 25% intensity

# Display
cv2.imshow("Fiber Zones", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()