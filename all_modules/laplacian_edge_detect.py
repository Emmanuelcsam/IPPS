#!/usr/bin/env python3
"""Apply Laplacian edge detection"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Laplacian
laplacian = np.abs(cv2.Laplacian(img, cv2.CV_64F))

# Normalize
if laplacian.max() > 0:
    laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)

cv2.imwrite('laplacian_edges.png', laplacian)