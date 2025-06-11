#!/usr/bin/env python3
"""Save intensity matrix as CSV file"""

import cv2
import numpy as np

# Image path
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"

# Load image as grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Failed to load image")
else:
    # Save as CSV with comma delimiter
    np.savetxt("intensity_matrix.csv", img, delimiter=',', fmt='%d')
    print(f"Saved intensity matrix to intensity_matrix.csv")
    print(f"Matrix shape: {img.shape}")