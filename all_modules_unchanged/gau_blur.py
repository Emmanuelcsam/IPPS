#!/usr/bin/env python3
"""Apply Gaussian blur to smooth difference map"""
import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create edge map
edges = cv2.Canny(img, 50, 150)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(edges, (7, 7), 0)

cv2.imwrite('blurred_edges.png', blurred)