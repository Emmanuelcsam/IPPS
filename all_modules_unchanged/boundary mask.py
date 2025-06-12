#!/usr/bin/env python3
"""Create boundary exclusion mask"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Parameters
center = (image.shape[1]//2, image.shape[0]//2)  # (x, y)
core_radius = 8
cladding_radius = 80
boundary_width = 3

# Create exclusion mask
h, w = image.shape[:2]
exclusion_mask = np.zeros((h, w), dtype=np.uint8)

# Core-cladding boundary
cv2.circle(exclusion_mask, center, 
          int(core_radius + boundary_width), 255, -1)
cv2.circle(exclusion_mask, center, 
          int(max(0, core_radius - boundary_width)), 0, -1)

# Cladding outer boundary
cv2.circle(exclusion_mask, center, 
          int(cladding_radius + boundary_width), 255, -1)
cv2.circle(exclusion_mask, center, 
          int(max(0, cladding_radius - boundary_width)), 0, -1)

# Display
cv2.imshow("Boundary Exclusion", exclusion_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()