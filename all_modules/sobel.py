#!/usr/bin/env python3
"""Apply Sobel edge detection"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Sobel gradients
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize
if sobel_mag.max() > 0:
    sobel_mag = (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)

cv2.imwrite('sobel_edges.png', sobel_mag)