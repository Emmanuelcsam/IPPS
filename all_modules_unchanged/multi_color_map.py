#!/usr/bin/env python3
"""Create custom multi-color gradient heatmap"""
import cv2
import numpy as np

# Load image and create edge map
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 30, 100)

# Create color gradient (black -> red -> yellow)
heatmap = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

# Red channel increases linearly
heatmap[:, :, 2] = edges

# Green channel increases for brighter values (creates yellow)
bright_mask = edges > 128
heatmap[bright_mask, 1] = ((edges[bright_mask] - 128) * 2).astype(np.uint8)

cv2.imwrite('multicolor_heatmap.png', heatmap)