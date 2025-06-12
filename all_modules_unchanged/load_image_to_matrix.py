#!/usr/bin/env python3
"""Load image and convert to intensity matrix"""

import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)

# Convert to grayscale intensity matrix
if len(img.shape) == 3:
    intensity_matrix = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    intensity_matrix = img

# Save as numpy array
np.save('intensity_matrix.npy', intensity_matrix)

print(f"Image shape: {intensity_matrix.shape}")
print(f"Intensity range: [{intensity_matrix.min()}, {intensity_matrix.max()}]")