#!/usr/bin/env python3
"""Apply a circular mask to an image"""
import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Create circular mask
center = (image.shape[1]//2, image.shape[0]//2)
radius = 100

h, w = image.shape[:2]
Y, X = np.ogrid[:h, :w]
dist_sq = (X - center[0])**2 + (Y - center[1])**2
mask = (dist_sq <= radius**2).astype(np.uint8)

# Apply mask to image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display
cv2.imshow("Original", image)
cv2.imshow("Masked", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()