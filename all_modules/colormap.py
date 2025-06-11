#!/usr/bin/env python3
"""Apply OpenCV heat colormap"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create edge map
edges = cv2.Canny(img, 50, 150)

# Apply heat colormap
heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)

cv2.imwrite('heat_colormap.png', heatmap)