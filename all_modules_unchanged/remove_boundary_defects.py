#!/usr/bin/env python3
"""
Remove defects near boundaries/edges
"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create defect mask
_, defect_mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Create boundary mask (example: circular fiber boundary)
h, w = image.shape
center = (w // 2, h // 2)
radius = min(h, w) // 2 - 20

# Create circular mask
fiber_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(fiber_mask, center, radius, 255, -1)

# Create boundary exclusion zone
boundary_width = 5
eroded_mask = cv2.erode(fiber_mask, np.ones((boundary_width*2, boundary_width*2), np.uint8))

# Boundary region is the difference
boundary_region = cv2.subtract(fiber_mask, eroded_mask)

# Remove defects in boundary region
cleaned_mask = cv2.bitwise_and(defect_mask, cv2.bitwise_not(boundary_region))

# Count removed defects
original_defects = np.sum(defect_mask > 0)
cleaned_defects = np.sum(cleaned_mask > 0)
removed_defects = original_defects - cleaned_defects

print(f"Original defect pixels: {original_defects}")
print(f"Cleaned defect pixels: {cleaned_defects}")
print(f"Removed boundary defects: {removed_defects}")

# Visualize
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
result[boundary_region > 0] = [255, 255, 0]  # Yellow for boundary
result[defect_mask > 0] = [0, 255, 255]  # Cyan for original defects
result[cleaned_mask > 0] = [0, 0, 255]  # Red for kept defects

cv2.imshow('Original Image', image)
cv2.imshow('Boundary Region', boundary_region)
cv2.imshow('Original Defects', defect_mask)
cv2.imshow('Cleaned Defects', cleaned_mask)
cv2.imshow('Result Visualization', result)
cv2.waitKey(0)
cv2.destroyAllWindows()