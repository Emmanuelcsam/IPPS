#!/usr/bin/env python3
"""Normalize intensity values to 0-255 range"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Normalize to 0-255 range
min_val = img.min()
max_val = img.max()

if max_val > min_val:
    normalized = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
else:
    normalized = img.astype(np.uint8)

# Save normalized image
cv2.imwrite('normalized_image.png', normalized)

print(f"Original range: [{min_val}, {max_val}]")
print(f"Normalized range: [0, 255]")