#!/usr/bin/env python3
"""Create black-to-red heatmap from difference map"""
import cv2
import numpy as np

# Load grayscale difference map
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create simple edge map
edges = cv2.Canny(img, 50, 150)

# Create BGR heatmap (black to red)
heatmap = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
heatmap[:, :, 2] = edges  # Red channel in BGR

cv2.imwrite('black_to_red_heatmap.png', heatmap)