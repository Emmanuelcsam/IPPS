#!/usr/bin/env python3
"""Create a circular mask"""
import cv2
import numpy as np

# Load image to get dimensions
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Create circular mask
center = (image.shape[1]//2, image.shape[0]//2)  # (x, y) center of image
radius = 100  # adjust as needed

h, w = image.shape[:2]
Y, X = np.ogrid[:h, :w]
dist_sq = (X - center[0])**2 + (Y - center[1])**2
mask = (dist_sq <= radius**2).astype(np.uint8) * 255

# Display mask
cv2.imshow("Circular Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()