#!/usr/bin/env python3
"""Find connected regions in difference map"""
import cv2
import numpy as np

# Load image and create binary edge map
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 150)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

# Create colored output
output = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for i in range(1, num_labels):  # Skip background (0)
    mask = labels == i
    # Random color for each component
    color = np.random.randint(0, 255, 3).tolist()
    output[mask] = color

cv2.imwrite('connected_components.png', output)
print(f"Found {num_labels - 1} connected regions")