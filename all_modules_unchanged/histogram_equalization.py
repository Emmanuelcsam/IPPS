#!/usr/bin/env python3
"""Apply histogram equalization for contrast enhancement"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized = cv2.equalizeHist(img)

# Save enhanced image
cv2.imwrite('histogram_equalized.png', equalized)

print("Applied histogram equalization for contrast enhancement")