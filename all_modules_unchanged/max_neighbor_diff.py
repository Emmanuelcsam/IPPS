#!/usr/bin/env python3
"""Find maximum difference to any neighboring pixel"""
import cv2
import numpy as np

# Load grayscale image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# 8-connected neighbor offsets
offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]

# Pad image for boundary handling
padded = np.pad(img, 1, mode='edge')
height, width = img.shape

# Find max difference to any neighbor
max_diff = np.zeros_like(img)
for dy, dx in offsets:
    neighbor = padded[1+dy:height+1+dy, 1+dx:width+1+dx]
    diff = np.abs(img - neighbor)
    max_diff = np.maximum(max_diff, diff)

# Normalize
if max_diff.max() > 0:
    max_diff = (max_diff / max_diff.max() * 255).astype(np.uint8)

cv2.imwrite('max_neighbor_diff.png', max_diff)