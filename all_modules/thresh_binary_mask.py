#!/usr/bin/env python3
"""Create binary mask from threshold"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create edge map
edges = cv2.Canny(img, 50, 150)

# Apply threshold (keep only strong edges)
threshold = 128
_, binary = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)

cv2.imwrite('threshold_binary.png', binary)