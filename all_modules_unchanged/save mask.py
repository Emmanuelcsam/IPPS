#!/usr/bin/env python3
"""Create and save a mask"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Create mask
center = (image.shape[1]//2, image.shape[0]//2)
radius = 80

h, w = image.shape[:2]
Y, X = np.ogrid[:h, :w]
dist_sq = (X - center[0])**2 + (Y - center[1])**2
mask = (dist_sq <= radius**2).astype(np.uint8) * 255

# Save mask
output_path = "fiber_mask.png"
cv2.imwrite(output_path, mask)
print(f"Mask saved to: {output_path}")