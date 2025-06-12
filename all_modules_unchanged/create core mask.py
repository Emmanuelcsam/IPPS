#!/usr/bin/env python3
"""Create core mask for fiber optic"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Core parameters
core_center = (image.shape[1]//2, image.shape[0]//2)  # (x, y)
core_radius = 8  # typical fiber core radius in pixels

# Create core mask
h, w = image.shape[:2]
Y, X = np.ogrid[:h, :w]
dist_sq = (X - core_center[0])**2 + (Y - core_center[1])**2
core_mask = (dist_sq <= core_radius**2).astype(np.uint8) * 255

# Display
cv2.imshow("Core Mask", core_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()